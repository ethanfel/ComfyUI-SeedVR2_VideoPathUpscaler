"""Focused media test for Direct Video's audio-preserving final file."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types
from fractions import Fraction

import av
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_direct_module():
    """Load the media module without importing the large SeedVR2 model graph."""
    src = types.ModuleType("src")
    src.__path__ = [str(ROOT / "src")]
    interfaces = types.ModuleType("src.interfaces")
    interfaces.__path__ = [str(ROOT / "src" / "interfaces")]
    sys.modules["src"] = src
    sys.modules["src.interfaces"] = interfaces

    video_upscaler = types.ModuleType("src.interfaces.video_upscaler")
    video_upscaler.SeedVR2VideoUpscaler = type(
        "SeedVR2VideoUpscaler", (), {"execute": None})
    sys.modules[video_upscaler.__name__] = video_upscaler

    cache = types.ModuleType("src.core.model_cache")
    cache.get_global_cache = lambda: None
    sys.modules[cache.__name__] = cache
    memory = types.ModuleType("src.optimization.memory_manager")
    memory.get_device_list = lambda **kwargs: ["cpu"]
    sys.modules[memory.__name__] = memory
    constants = types.ModuleType("src.utils.constants")
    constants.__version__ = "test"
    sys.modules[constants.__name__] = constants
    debug = types.ModuleType("src.utils.debug")
    debug.Debug = object
    sys.modules[debug.__name__] = debug

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = tempfile.gettempdir
    sys.modules["folder_paths"] = folder_paths
    latest = types.ModuleType("comfy_api.latest")
    latest.Input = types.SimpleNamespace(Video=object)
    latest.InputImpl = object
    latest.io = types.SimpleNamespace(ComfyNode=object)
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    spec = importlib.util.spec_from_file_location(
        "src.interfaces.direct_video_upscaler",
        ROOT / "src" / "interfaces" / "direct_video_upscaler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mux_audio_into_direct_video_output():
    module = load_direct_module()
    with tempfile.TemporaryDirectory() as temporary:
        silent = str(pathlib.Path(temporary, "silent.mp4"))
        final = str(pathlib.Path(temporary, "final.mp4"))
        writer = module._VideoWriter(silent, Fraction(24, 1), 18)
        writer.write(torch.linspace(0.0, 1.0, 4).reshape(4, 1, 1, 1).expand(
            4, 16, 16, 3).clone())
        writer.close()
        samples = round(4 / 24 * 8000)
        module.mux_audio_into_video(
            silent, {
                "waveform": torch.zeros((1, 2, samples), dtype=torch.float32),
                "sample_rate": 8000,
            }, final)

        with av.open(final, mode="r") as container:
            assert len(container.streams.video) == 1
            assert len(container.streams.audio) == 1
            assert sum(1 for _ in container.decode(container.streams.video[0])) == 4
        assert pathlib.Path(final).stat().st_size > 0


if __name__ == "__main__":
    test_mux_audio_into_direct_video_output()
    print("direct-video embedded audio unit test passed")
