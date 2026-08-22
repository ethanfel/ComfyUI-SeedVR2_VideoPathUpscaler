"""Collision-free native-video path for SeedVR2 in ComfyUI."""

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import SeedVR2VideoPathExtension, comfy_entrypoint

__all__ = ["comfy_entrypoint", "SeedVR2VideoPathExtension"]
