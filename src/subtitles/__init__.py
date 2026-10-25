"""Subtitle generation and transcription module for ClipForge."""

from src.subtitles.base import (
    BaseTranscriptionProvider,
    SubtitleSegment,
    TranscriptionResult,
    WordTimestamp,
    format_timestamp_srt,
)
from src.subtitles.burner import SubtitleBurnError, burn_subtitles_to_video
from src.subtitles.registry import transcription_registry
from src.subtitles.whisper_provider import FasterWhisperProvider

__all__ = [
    "BaseTranscriptionProvider",
    "FasterWhisperProvider",
    "SubtitleBurnError",
    "SubtitleSegment",
    "TranscriptionResult",
    "WordTimestamp",
    "burn_subtitles_to_video",
    "format_timestamp_srt",
    "transcription_registry",
]
