"""Public ComfyUI registry for the SeedVR2 native-video path."""

from comfy_api.latest import ComfyExtension, io

from .auto_configurator import SeedVR2VideoPathAutoConfigurator
from .direct_video_upscaler import SeedVR2VideoPathUpscaler


class SeedVR2VideoPathExtension(ComfyExtension):
    """SeedVR2 native-video extension with collision-free public node IDs."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return only the nodes owned by this native-video add-on."""
        return [
            SeedVR2VideoPathUpscaler,
            SeedVR2VideoPathAutoConfigurator,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    """Return the ComfyUI V3 extension entry point."""
    return SeedVR2VideoPathExtension()


__all__ = [
    "SeedVR2VideoPathUpscaler",
    "SeedVR2VideoPathAutoConfigurator",
    "SeedVR2VideoPathExtension",
    "comfy_entrypoint",
]
