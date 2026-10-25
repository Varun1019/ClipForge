"""Video reframing and aspect ratio conversion for ClipForge."""

from src.framing.options import (
    AspectRatioMode,
    FramingConfig,
    FramingStyle,
    SQUARE_RESOLUTIONS,
    VERTICAL_RESOLUTIONS,
)
from src.framing.reframe import FramingError, build_reframe_filter, reframe_video

__all__ = [
    "AspectRatioMode",
    "FramingConfig",
    "FramingStyle",
    "FramingError",
    "SQUARE_RESOLUTIONS",
    "VERTICAL_RESOLUTIONS",
    "build_reframe_filter",
    "reframe_video",
]
