"""Abstract base interfaces and data structures for ClipForge subtitle and transcription subsystem."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


def format_timestamp_srt(seconds: float) -> str:
    """
    Format floating seconds into standard SubRip (SRT) timestamp: HH:MM:SS,mmm.
    Clamps negative values to zero.
    """
    safe_secs = max(0.0, float(seconds))
    hrs = int(safe_secs // 3600)
    rem = safe_secs % 3600
    mins = int(rem // 60)
    secs = int(rem % 60)
    millis = int(round((rem - int(rem)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    if secs >= 60:
        mins += 1
        secs -= 60
    if mins >= 60:
        hrs += 1
        mins -= 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


@dataclass
class WordTimestamp:
    """Represents a single transcribed word with exact boundary timestamps."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class SubtitleSegment:
    """Represents a coherent spoken subtitle segment or phrase."""
    id: int
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def shift(self, offset_secs: float) -> "SubtitleSegment":
        """Return a copy of the segment with start and end timestamps shifted by offset_secs."""
        shifted_words = [
            WordTimestamp(
                word=w.word,
                start=max(0.0, w.start + offset_secs),
                end=max(0.0, w.end + offset_secs),
                probability=w.probability,
            )
            for w in self.words
        ]
        return SubtitleSegment(
            id=self.id,
            start=max(0.0, self.start + offset_secs),
            end=max(0.0, self.end + offset_secs),
            text=self.text,
            words=shifted_words,
        )


@dataclass
class TranscriptionResult:
    """Encapsulates the complete transcription result for a media track."""
    segments: list[SubtitleSegment]
    language: str = "en"
    duration: float = 0.0

    @property
    def has_speech(self) -> bool:
        """True if one or more valid text segments were detected."""
        return len(self.segments) > 0 and any(s.text.strip() for s in self.segments)

    def to_srt(self, offset_secs: float = 0.0) -> str:
        """
        Convert transcription segments into standard SRT format.
        Optionally shifts timestamps by offset_secs (e.g. to re-align with clip offset).
        """
        if not self.has_speech:
            return ""

        srt_blocks: list[str] = []
        for idx, seg in enumerate(self.segments, 1):
            s_start = max(0.0, seg.start + offset_secs)
            s_end = max(s_start + 0.1, seg.end + offset_secs)
            clean_text = seg.text.strip()
            if not clean_text:
                continue

            time_header = f"{format_timestamp_srt(s_start)} --> {format_timestamp_srt(s_end)}"
            srt_blocks.append(f"{idx}\n{time_header}\n{clean_text}\n")

        return "\n".join(srt_blocks)


class BaseTranscriptionProvider(ABC):
    """Abstract interface for speech transcription engines in ClipForge."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for the transcription provider (e.g. 'faster-whisper')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """User-friendly name for UI presentation."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if dependencies and environment are ready for inference."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_or_video_path: Path,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe the dialogue of an audio or video file.
        Must handle silent media gracefully without raising errors.
        """
        pass
