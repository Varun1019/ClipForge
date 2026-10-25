"""Aspect ratio and video framing configuration for ClipForge.

Supports original ratio, vertical (9:16) for Shorts/Reels/TikTok,
and square (1:1) for Instagram/LinkedIn feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class AspectRatioMode(str, Enum):
    ORIGINAL = "original"
    VERTICAL_9_16 = "9:16"
    SQUARE_1_1 = "1:1"

    def label(self) -> str:
        if self == AspectRatioMode.ORIGINAL:
            return "Original (Preserve Source Ratio)"
        if self == AspectRatioMode.VERTICAL_9_16:
            return "Vertical 9:16 (Shorts / Reels / TikTok)"
        if self == AspectRatioMode.SQUARE_1_1:
            return "Square 1:1 (Feed / Post)"
        return self.value


class FramingStyle(str, Enum):
    BLURRED_BACKGROUND = "blur"
    CENTER_CROP = "crop"

    def label(self) -> str:
        if self == FramingStyle.BLURRED_BACKGROUND:
            return "Blurred Background (Fit Entire Frame)"
        if self == FramingStyle.CENTER_CROP:
            return "Center Crop (Fill Canvas)"
        return self.value


VERTICAL_RESOLUTIONS: dict[str, Tuple[int, int]] = {
    "1080x1920 (Full HD 9:16)": (1080, 1920),
    "720x1280 (HD 9:16)": (720, 1280),
}

SQUARE_RESOLUTIONS: dict[str, Tuple[int, int]] = {
    "1080x1080 (Full HD 1:1)": (1080, 1080),
    "720x720 (Standard 1:1)": (720, 720),
}


@dataclass(frozen=True)
class FramingConfig:
    aspect_ratio: AspectRatioMode = AspectRatioMode.ORIGINAL
    style: FramingStyle = FramingStyle.BLURRED_BACKGROUND
    target_width: int = 1080
    target_height: int = 1920

    def is_reframe_needed(self) -> bool:
        return self.aspect_ratio != AspectRatioMode.ORIGINAL

    def get_suffix(self) -> str:
        if self.aspect_ratio == AspectRatioMode.VERTICAL_9_16:
            return "_vertical"
        if self.aspect_ratio == AspectRatioMode.SQUARE_1_1:
            return "_square"
        return ""

    def validate(self) -> None:
        if not self.is_reframe_needed():
            return
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError(
                f"Target dimensions must be positive, got {self.target_width}x{self.target_height}"
            )
        if self.target_width > 7680 or self.target_height > 4320:
            raise ValueError(
                f"Target dimensions exceed maximum 8K limit, got {self.target_width}x{self.target_height}"
            )
        if self.target_width % 2 != 0 or self.target_height % 2 != 0:
            raise ValueError(
                f"Target dimensions must be even for H.264 encoding, got {self.target_width}x{self.target_height}"
            )
