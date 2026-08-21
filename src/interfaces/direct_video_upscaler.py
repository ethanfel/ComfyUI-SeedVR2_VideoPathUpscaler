"""File-backed SeedVR2 video upscaling for ComfyUI's native VIDEO type."""

from __future__ import annotations

import io as stdlib_io
import math
import os
import uuid
from fractions import Fraction
from typing import Any, Dict, Generator, Optional

import av
import numpy as np
import torch
import folder_paths
from comfy_api.latest import Input, InputImpl, io

from .video_upscaler import SeedVR2VideoUpscaler
from ..core.model_cache import get_global_cache
from ..optimization.memory_manager import get_device_list
from ..utils.constants import __version__
from ..utils.debug import Debug


def _rewind(source: str | stdlib_io.BytesIO) -> None:
    if isinstance(source, stdlib_io.BytesIO):
        source.seek(0)


def _frame_timestamp(frame: av.VideoFrame, stream: av.VideoStream, fallback: float) -> float:
    if frame.pts is not None:
        time_base = frame.time_base or stream.time_base
        if time_base is not None:
            return float(frame.pts * time_base)
    return fallback


def stream_video_frame_chunks(
    source: str | stdlib_io.BytesIO,
    start_time: float,
    duration: float,
    chunk_size: int,
) -> Generator[torch.Tensor, None, None]:
    """Decode RGB frames incrementally without materializing the complete video."""
    _rewind(source)
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("Input VIDEO has no decodable video stream")
        stream = container.streams.video[0]
        frame_rate = float(stream.average_rate) if stream.average_rate else 30.0
        frame_period = 1.0 / max(frame_rate, 1e-6)
        end_time = start_time + duration if duration > 0 else None

        if start_time > 0:
            container.seek(int(start_time * av.time_base), backward=True)

        frames = []
        fallback_time = 0.0
        for frame in container.decode(stream):
            timestamp = _frame_timestamp(frame, stream, fallback_time)
            fallback_time = timestamp + frame_period
            if timestamp + (frame_period * 0.5) < start_time:
                continue
            if end_time is not None and timestamp >= end_time:
                break

            array = frame.to_ndarray(format="rgb24")
            rotation = getattr(frame, "rotation", 0) or 0
            if rotation:
                array = np.rot90(array, k=int(round(rotation / 90)), axes=(0, 1)).copy()
            frames.append(torch.from_numpy(array))
            if len(frames) == chunk_size:
                yield torch.stack(frames).to(dtype=torch.float32).div_(255.0)
                frames.clear()

        if frames:
            yield torch.stack(frames).to(dtype=torch.float32).div_(255.0)


def extract_full_audio(
    source: str | stdlib_io.BytesIO,
    start_time: float,
    duration: float,
) -> Optional[Dict[str, Any]]:
    """Decode only the active video's complete audio window into one Comfy AUDIO value."""
    _rewind(source)
    with av.open(source, mode="r") as container:
        audio_stream = next(
            (stream for stream in reversed(container.streams.audio) if stream.codec_context is not None),
            None,
        )
        if audio_stream is None:
            return None

        if start_time > 0:
            container.seek(int(start_time * av.time_base), backward=True)

        end_time = start_time + duration if duration > 0 else None
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        chunks = []
        sample_rate = int(audio_stream.sample_rate or 0)
        fallback_time = 0.0
        done = False

        for packet in container.demux(audio_stream):
            try:
                decoded_frames = packet.decode()
            except av.error.FFmpegError:
                continue
            for decoded in decoded_frames:
                for frame in resampler.resample(decoded):
                    rate = int(frame.sample_rate or sample_rate or 1)
                    sample_rate = rate
                    if frame.pts is not None and frame.time_base is not None:
                        frame_start = float(frame.pts * frame.time_base)
                    else:
                        frame_start = fallback_time
                    frame_end = frame_start + (frame.samples / rate)
                    fallback_time = frame_end

                    if frame_end <= start_time:
                        continue
                    if end_time is not None and frame_start >= end_time:
                        done = True
                        break

                    left = max(0, int(math.ceil((start_time - frame_start) * rate)))
                    right = frame.samples
                    if end_time is not None:
                        right = min(right, int(math.ceil((end_time - frame_start) * rate)))
                    if right > left:
                        chunks.append(frame.to_ndarray()[..., left:right])
                if done:
                    break
            if done:
                break

        if not chunks:
            return None
        waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0)
        if duration > 0:
            waveform = waveform[..., :int(math.ceil(duration * sample_rate))]
        return {"waveform": waveform, "sample_rate": sample_rate}


def mux_audio_into_video(
    video_path: str,
    audio: Dict[str, Any],
    output_path: str,
) -> None:
    """Stream-copy H.264 and embed the complete Comfy AUDIO track as AAC."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not torch.is_tensor(waveform) or sample_rate <= 0:
        raise ValueError("Audio must contain a tensor waveform and positive sample_rate")
    if waveform.ndim == 3:
        if int(waveform.shape[0]) != 1:
            raise ValueError("Direct-video audio must contain exactly one batch")
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or int(waveform.shape[-1]) < 1:
        raise ValueError("Direct-video audio must be [channels,samples]")
    channels = int(waveform.shape[0])
    layouts = {
        1: "mono",
        2: "stereo",
        3: "2.1",
        4: "quad",
        5: "5.0",
        6: "5.1",
        8: "7.1",
    }
    layout = layouts.get(channels)
    if layout is None:
        raise ValueError(f"Cannot embed an unsupported {channels}-channel audio layout")
    samples = (
        waveform.detach().to(device="cpu", dtype=torch.float32)
        .clamp_(-1.0, 1.0).contiguous().numpy()
    )

    source = output = None
    try:
        source = av.open(video_path, mode="r")
        if not source.streams.video:
            raise ValueError("Direct-video temporary file has no video stream")
        input_video = source.streams.video[0]
        output = av.open(
            output_path,
            mode="w",
            options={"movflags": "use_metadata_tags+faststart"},
        )
        for key, value in source.metadata.items():
            output.metadata[str(key)] = str(value)
        output_video = output.add_stream_from_template(input_video)
        output_audio = output.add_stream("aac", rate=sample_rate)
        output_audio.bit_rate = 256_000
        output_audio.layout = layout

        for packet in source.demux(input_video):
            if packet.dts is None:
                continue
            packet.stream = output_video
            output.mux(packet)

        for start in range(0, int(samples.shape[-1]), 1024):
            stop = min(int(samples.shape[-1]), start + 1024)
            frame = av.AudioFrame.from_ndarray(
                samples[:, start:stop], format="fltp", layout=layout
            )
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in output_audio.encode(frame):
                output.mux(packet)
        for packet in output_audio.encode():
            output.mux(packet)

        output.close()
        output = None
        source.close()
        source = None
    except BaseException:
        if output is not None:
            output.close()
        if source is not None:
            source.close()
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


class _VideoWriter:
    """Small streaming H.264 writer used to keep the node's output file-backed."""

    def __init__(self, path: str, frame_rate: Fraction, crf: int):
        self.path = path
        self.frame_rate = frame_rate
        self.crf = crf
        self.container = None
        self.stream = None

    def write(self, frames: torch.Tensor) -> None:
        if frames.ndim != 4 or frames.shape[-1] < 3:
            raise ValueError(f"Expected RGB frames [T,H,W,C], got {tuple(frames.shape)}")
        if self.container is None:
            height, width = int(frames.shape[1]), int(frames.shape[2])
            self.container = av.open(
                self.path,
                mode="w",
                options={"movflags": "use_metadata_tags+faststart"},
            )
            self.stream = self.container.add_stream("libx264", rate=self.frame_rate)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": str(self.crf), "preset": "medium"}

        frames = frames[..., :3].detach().to(device="cpu", dtype=torch.float32)
        frames = frames.clamp_(0.0, 1.0).mul_(255.0).to(dtype=torch.uint8)
        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame.contiguous().numpy(), format="rgb24")
            for packet in self.stream.encode(av_frame):
                self.container.mux(packet)

    def close(self) -> None:
        if self.container is None:
            return
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        self.container = None
        self.stream = None


class SeedVR2DirectVideoUpscaler(io.ComfyNode):
    """Stream a native VIDEO through SeedVR2 and return a file-backed VIDEO plus full AUDIO."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeedVR2DirectVideoUpscaler",
            display_name=f"SeedVR2 Direct Video Upscaler (v{__version__})",
            category="SEEDVR2",
            description=(
                "Upscales a native ComfyUI VIDEO in bounded chunks. The complete frame sequence "
                "never becomes an IMAGE tensor in the workflow; the output remains file-backed. "
                "Source audio is preserved in the returned VIDEO and is also returned once as AUDIO."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip=(
                        "Native file-backed VIDEO, normally from Load Video or "
                        "MiniMax H3 Full-Chain Latent Video Adapter."
                    ),
                ),
                io.Custom("SEEDVR2_DIT").Input("dit"),
                io.Custom("SEEDVR2_VAE").Input("vae"),
                io.Int.Input("seed", default=42, min=0, max=2**32 - 1, step=1),
                io.Int.Input("resolution", default=1080, min=16, max=16384, step=2),
                io.Int.Input("max_resolution", default=0, min=0, max=16384, step=2),
                io.Int.Input(
                    "batch_size", default=5, min=1, max=16384, step=4,
                    tooltip="SeedVR2 temporal batch size; values following 4n+1 are recommended.",
                ),
                io.Int.Input(
                    "chunk_size", default=21, min=1, max=4096, step=1,
                    tooltip="Maximum source frames decoded and held in RAM at once.",
                ),
                io.Int.Input(
                    "chunk_overlap", default=2, min=0, max=32, step=1, optional=True,
                    tooltip="Raw context frames carried from the prior file chunk and removed from output.",
                ),
                io.Boolean.Input("uniform_batch_size", default=False, optional=True),
                io.Int.Input("temporal_overlap", default=0, min=0, max=16, step=1, optional=True),
                io.Int.Input("prepend_frames", default=0, min=0, max=32, step=1, optional=True),
                io.Combo.Input(
                    "color_correction",
                    options=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    default="lab",
                ),
                io.Float.Input("input_noise_scale", default=0.0, min=0.0, max=1.0, step=0.001, optional=True),
                io.Float.Input("latent_noise_scale", default=0.0, min=0.0, max=1.0, step=0.001, optional=True),
                io.Combo.Input(
                    "offload_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="cpu",
                    optional=True,
                ),
                io.Int.Input(
                    "temporary_video_crf", default=18, min=0, max=51, step=1, optional=True,
                    tooltip="H.264 quality of the file-backed intermediate; lower is higher quality.",
                ),
                io.Boolean.Input("enable_debug", default=False, optional=True),
            ],
            outputs=[
                io.Video.Output(display_name="video", tooltip="File-backed upscaled video with source audio preserved when present."),
                io.Audio.Output(display_name="audio", tooltip="Full unchunked audio from the active source video window."),
            ],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        dit: Dict[str, Any],
        vae: Dict[str, Any],
        seed: int,
        resolution: int = 1080,
        max_resolution: int = 0,
        batch_size: int = 5,
        chunk_size: int = 21,
        chunk_overlap: int = 2,
        uniform_batch_size: bool = False,
        temporal_overlap: int = 0,
        prepend_frames: int = 0,
        color_correction: str = "lab",
        input_noise_scale: float = 0.0,
        latent_noise_scale: float = 0.0,
        offload_device: str = "cpu",
        temporary_video_crf: int = 18,
        enable_debug: bool = False,
    ) -> io.NodeOutput:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        source = video.get_stream_source()
        start_time, duration = video.get_active_trim_window()
        frame_rate = Fraction(video.get_frame_rate())
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"seedvr2_direct_{uuid.uuid4().hex}.mp4")
        silent_path = os.path.join(temp_dir, f"seedvr2_direct_{uuid.uuid4().hex}.silent.mp4")

        run_id = uuid.uuid4().hex
        chunk_dit = dict(dit)
        chunk_vae = dict(vae)
        temporary_dit_cache = not bool(chunk_dit.get("cache_model", False))
        temporary_vae_cache = not bool(chunk_vae.get("cache_model", False))
        if temporary_dit_cache:
            chunk_dit.update(cache_model=True, node_id=f"seedvr2-direct-{run_id}-dit")
        if temporary_vae_cache:
            chunk_vae.update(cache_model=True, node_id=f"seedvr2-direct-{run_id}-vae")

        debug = Debug(enabled=enable_debug)
        writer = _VideoWriter(silent_path, frame_rate, temporary_video_crf)
        previous_tail = None
        frames_written = 0
        chunk_index = 0

        try:
            for new_frames in stream_video_frame_chunks(
                source, start_time, duration, chunk_size
            ):
                chunk_index += 1
                if previous_tail is not None and chunk_overlap:
                    context_count = min(chunk_overlap, previous_tail.shape[0])
                    input_frames = torch.cat((previous_tail[-context_count:], new_frames), dim=0)
                else:
                    context_count = 0
                    input_frames = new_frames

                debug.log(
                    f"Direct-video chunk {chunk_index}: {new_frames.shape[0]} new + "
                    f"{context_count} context frames",
                    category="video",
                    force=True,
                )
                node_output = SeedVR2VideoUpscaler.execute(
                    image=input_frames,
                    dit=chunk_dit,
                    vae=chunk_vae,
                    seed=seed,
                    resolution=resolution,
                    max_resolution=max_resolution,
                    batch_size=batch_size,
                    uniform_batch_size=uniform_batch_size,
                    temporal_overlap=temporal_overlap,
                    prepend_frames=prepend_frames if chunk_index == 1 else 0,
                    color_correction=color_correction,
                    input_noise_scale=input_noise_scale,
                    latent_noise_scale=latent_noise_scale,
                    offload_device=offload_device,
                    enable_debug=enable_debug,
                )
                result = node_output[0]
                if context_count:
                    result = result[context_count:]
                writer.write(result)
                frames_written += int(result.shape[0])
                previous_tail = new_frames[-chunk_overlap:].clone() if chunk_overlap else None
                del input_frames, result, node_output, new_frames

            if frames_written == 0:
                raise ValueError("No frames were decoded from the active VIDEO window")
            writer.close()

            # Decode audio once, after model processing, so it is not held in RAM for the run.
            audio = extract_full_audio(source, start_time, duration)
            if audio is None:
                os.replace(silent_path, output_path)
            else:
                mux_audio_into_video(silent_path, audio, output_path)
                os.remove(silent_path)
            return io.NodeOutput(InputImpl.VideoFromFile(output_path), audio)
        except BaseException:
            writer.close()
            for path in (silent_path, output_path):
                if os.path.exists(path):
                    os.remove(path)
            raise
        finally:
            cache = get_global_cache()
            if temporary_dit_cache:
                cache.remove_dit(chunk_dit, debug)
            if temporary_vae_cache:
                cache.remove_vae(chunk_vae, debug)


__all__ = [
    "SeedVR2DirectVideoUpscaler",
    "extract_full_audio",
    "mux_audio_into_video",
    "stream_video_frame_chunks",
]
