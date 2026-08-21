"""ComfyUI node for hardware-aware SeedVR2 configuration."""

from __future__ import annotations

from typing import Optional

import av
import torch
from comfy_api.latest import Input, io
from comfy_execution.utils import get_executing_context

from ..core.auto_config import recommend_seedvr2_config
from ..optimization.compatibility import COMFY_KITCHEN_INT8_ATTENTION_AVAILABLE
from ..optimization.memory_manager import get_device_list


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        best_index = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.get_device_properties(index).total_memory,
        )
        return torch.device(f"cuda:{best_index}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("SeedVR2 Auto Configurator could not find a supported GPU")


def _hardware_info(device: torch.device) -> tuple[str, float, float, tuple[int, int]]:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return (
            properties.name,
            total_bytes / (1024**3),
            free_bytes / (1024**3),
            (properties.major, properties.minor),
        )
    if device.type == "mps":
        import psutil

        memory = psutil.virtual_memory()
        total = memory.total / (1024**3)
        free = memory.available / (1024**3)
        return "Apple Silicon", total, free, (0, 0)
    raise RuntimeError(f"Unsupported auto-configuration device: {device}")


def _kitchen_attention_available(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        import comfy_kitchen

        return bool(comfy_kitchen.int8_attention_is_available(device))
    except (ImportError, AttributeError, OSError, RuntimeError):
        return COMFY_KITCHEN_INT8_ATTENTION_AVAILABLE


def _reliable_video_frame_count(video: Input.Video) -> Optional[int]:
    """Return an active frame count, correcting false one-frame VIDEO metadata."""
    reported_count = None
    try:
        reported_count = int(video.get_frame_count())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass

    # Avoid opening the stream when the VIDEO implementation already supplied
    # a useful count. Some file-backed implementations return 1 when their
    # container does not expose nb_frames and duration=0 means "until EOF".
    if reported_count is not None and reported_count > 1:
        return reported_count

    try:
        source = video.get_stream_source()
        start_time, requested_duration = video.get_active_trim_window()
        with av.open(source, mode="r") as container:
            if not container.streams.video:
                return reported_count
            stream = container.streams.video[0]
            frame_rate = stream.average_rate or video.get_frame_rate()
            frame_rate = float(frame_rate)
            if frame_rate <= 0:
                return reported_count

            raw_duration = None
            if stream.duration is not None and stream.time_base is not None:
                raw_duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                raw_duration = float(container.duration / av.time_base)
            elif stream.frames and stream.frames > 0:
                raw_duration = float(stream.frames / frame_rate)

            if raw_duration is None:
                return reported_count
            remaining_duration = max(0.0, raw_duration - float(start_time))
            active_duration = (
                min(float(requested_duration), remaining_duration)
                if requested_duration and requested_duration > 0
                else remaining_duration
            )
            estimated_count = int(round(active_duration * frame_rate))
            if estimated_count > 0:
                return estimated_count
    except (AttributeError, av.FFmpegError, OSError, RuntimeError, TypeError, ValueError):
        pass

    return reported_count


class SeedVR2AutoConfigurator(io.ComfyNode):
    """Detect hardware and emit complete model plus runtime configurations."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = ["auto", *get_device_list()]
        return io.Schema(
            node_id="SeedVR2AutoConfigurator",
            display_name="SeedVR2 Auto Configurator",
            category="SEEDVR2",
            description=(
                "Detects the selected GPU and configures model precision, attention, batching, "
                "tiling, model residency, tensor offload, and direct-video chunking. Connect all "
                "three configuration outputs to a SeedVR2 upscaler."
            ),
            inputs=[
                io.Combo.Input("device", options=devices, default="auto"),
                io.Combo.Input(
                    "profile",
                    options=["balanced", "maximum_quality", "maximum_throughput"],
                    default="balanced",
                ),
                io.Int.Input(
                    "target_resolution", default=1080, min=16, max=4320, step=2,
                    tooltip="Desired output short-edge resolution; hardware settings adapt around it.",
                ),
                io.Float.Input(
                    "reserve_vram_gb", default=4.0, min=0.0, max=64.0, step=0.5,
                    tooltip="VRAM kept free for ComfyUI, previews, and other nodes.",
                ),
                io.Int.Input(
                    "batch_size_override", default=0, min=0, max=16381, step=1, optional=True,
                    tooltip=(
                        "Override the recommended temporal batch size. 0 keeps automatic sizing. "
                        "Non-zero values must follow 4n+1: 1, 5, 9, 13, ..."
                    ),
                ),
                io.Boolean.Input(
                    "enable_torch_compile", default=False, optional=True,
                    tooltip="Enable only for repeated runs; the first run has significant compilation cost.",
                ),
                io.Video.Input(
                    "video", optional=True,
                    tooltip="Optional native VIDEO used to avoid oversized batches/chunks on short clips.",
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_DIT").Output(display_name="dit"),
                io.Custom("SEEDVR2_VAE").Output(display_name="vae"),
                io.Custom("SEEDVR2_AUTO_SETTINGS").Output(display_name="auto_settings"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        device: str,
        profile: str,
        target_resolution: int,
        reserve_vram_gb: float,
        batch_size_override: int = 0,
        enable_torch_compile: bool = False,
        video: Optional[Input.Video] = None,
    ) -> io.NodeOutput:
        selected_device = _resolve_device(device)
        gpu_name, total_gb, free_gb, capability = _hardware_info(selected_device)
        planning_free_gb = free_gb
        if selected_device.type == "cuda":
            # Existing SeedVR2 caches owned by this process are reclaimable if
            # the recommendation changes, so do not progressively downgrade on
            # every run merely because the previous model remains resident.
            planning_free_gb = min(
                total_gb,
                free_gb + torch.cuda.memory_allocated(selected_device) / (1024**3),
            )
        frame_count = None
        if video is not None:
            frame_count = _reliable_video_frame_count(video)

        recommendation = recommend_seedvr2_config(
            total_vram_gb=total_gb,
            free_vram_gb=planning_free_gb,
            compute_capability=capability,
            target_resolution=target_resolution,
            profile=profile,
            reserve_vram_gb=reserve_vram_gb,
            frame_count=frame_count,
            batch_size_override=batch_size_override or None,
            kitchen_attention_available=_kitchen_attention_available(selected_device),
            enable_torch_compile=enable_torch_compile,
        )

        node_id = get_executing_context().node_id
        dit = {
            **recommendation["dit"],
            "device": str(selected_device),
            "node_id": f"{node_id}:auto-dit",
        }
        vae = {
            **recommendation["vae"],
            "device": str(selected_device),
            "node_id": f"{node_id}:auto-vae",
        }
        settings = recommendation["runtime"]
        hardware = recommendation["hardware"]
        residency = "GPU-resident cache" if hardware["keep_models_on_gpu"] else "CPU model cache"
        chunk_description = "disabled" if settings["chunk_size"] == 0 else str(settings["chunk_size"])
        frame_description = f" | frames {frame_count}" if frame_count is not None else ""
        batch_description = str(settings["batch_size"])
        if recommendation["batch_size_overridden"]:
            batch_description += " (override)"
        report = (
            f"{gpu_name} | {total_gb:.1f} GB total, {free_gb:.1f} GB free, "
            f"{hardware['budget_gb']:.1f} GB usable | {dit['model']} | "
            f"batch {batch_description} | chunks {chunk_description}{frame_description} | "
            f"{residency} | attention {dit['attention_mode']} | "
            f"{recommendation['model_reason']}"
        )
        print(f"⚙️ SeedVR2 auto configuration: {report}", flush=True)
        return io.NodeOutput(dit, vae, settings, report)


__all__ = ["SeedVR2AutoConfigurator"]
