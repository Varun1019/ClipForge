"""ClipForge video metadata extraction and validation module."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Optional


class FFmpegNotFoundError(Exception):
    """Raised when FFmpeg or ffprobe executable is not available in system PATH."""
    pass


class InvalidVideoError(Exception):
    """Raised when the target file is corrupted, unreadable, or lacks a valid video stream."""
    pass


# In-memory cache for extracted metadata: key -> (mtime, size_bytes, metadata)
_METADATA_CACHE: dict[str, tuple[float, int, "VideoMetadata"]] = {}


def clear_metadata_cache() -> None:
    """Clear in-memory metadata cache."""
    _METADATA_CACHE.clear()


@dataclass
class VideoMetadata:
    """Structured metadata for an ingested video."""
    filename: str
    file_size_bytes: int
    file_size_formatted: str
    duration_seconds: float
    duration_formatted: str
    width: int
    height: int
    resolution_str: str
    codec: str
    format_name: str


def check_ffprobe_available() -> bool:
    """Check if ffprobe is installed and reachable via PATH."""
    return shutil.which("ffprobe") is not None


def format_file_size(size_bytes: int) -> str:
    """Format bytes into human-readable size string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to standard HH:MM:SS or MM:SS."""
    if seconds < 0:
        return "00:00"
    
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_video_metadata(
    file_path: Path | str,
    original_filename: Optional[str] = None
) -> VideoMetadata:
    """
    Extract video metadata (resolution, duration, size, codec) using ffprobe.

    Args:
        file_path: Path to the target video file.
        original_filename: Optional original upload name if file_path is temporary.

    Returns:
        VideoMetadata dataclass containing validated media information.

    Raises:
        FFmpegNotFoundError: If ffprobe binary is not found on system.
        InvalidVideoError: If file is corrupted, unreadable, or missing video stream.
        FileNotFoundError: If the specified file_path does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found at path: {path}")

    # Check cache based on resolved path, mtime, size, and original_filename
    stat = path.stat()
    cache_key = f"{path.resolve()}:{original_filename or ''}"
    cached = _METADATA_CACHE.get(cache_key)
    if cached is not None:
        cached_mtime, cached_size, cached_meta = cached
        if cached_mtime == stat.st_mtime and cached_size == stat.st_size:
            return cached_meta

    if not check_ffprobe_available():
        raise FFmpegNotFoundError(
            "FFmpeg/ffprobe is not available on this system. "
            "Please install FFmpeg to inspect and process video files."
        )

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,codec_name",
        "-show_entries", "format=duration,size,format_name",
        "-of", "json",
        str(path.resolve())
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise InvalidVideoError("ffprobe timed out while analyzing video file.") from exc
    except Exception as exc:
        raise InvalidVideoError(f"Unexpected error running ffprobe: {exc}") from exc

    if result.returncode != 0:
        err_msg = result.stderr.strip() or "Unknown ffprobe error"
        raise InvalidVideoError(f"Failed to inspect video: {err_msg}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidVideoError("Failed to parse ffprobe JSON output.") from exc

    streams = data.get("streams", [])
    if not streams:
        raise InvalidVideoError("No valid video stream found in the file.")

    video_stream = streams[0]
    width = video_stream.get("width")
    height = video_stream.get("height")

    if not width or not height or width <= 0 or height <= 0:
        raise InvalidVideoError("Invalid or missing video resolution in video stream.")

    # Duration: check format duration first, fallback to stream duration
    format_info = data.get("format", {})
    raw_duration = format_info.get("duration") or video_stream.get("duration")
    
    try:
        duration_seconds = float(raw_duration) if raw_duration is not None else 0.0
    except (ValueError, TypeError):
        duration_seconds = 0.0

    # File size: from format info or filesystem directly
    raw_size = format_info.get("size")
    if raw_size is not None:
        try:
            file_size_bytes = int(raw_size)
        except (ValueError, TypeError):
            file_size_bytes = path.stat().st_size
    else:
        file_size_bytes = path.stat().st_size

    codec = video_stream.get("codec_name", "unknown")
    format_name = format_info.get("format_name", "unknown")
    display_filename = original_filename or path.name

    meta = VideoMetadata(
        filename=display_filename,
        file_size_bytes=file_size_bytes,
        file_size_formatted=format_file_size(file_size_bytes),
        duration_seconds=duration_seconds,
        duration_formatted=format_duration(duration_seconds),
        width=int(width),
        height=int(height),
        resolution_str=f"{width} × {height}",
        codec=codec,
        format_name=format_name,
    )
    _METADATA_CACHE[cache_key] = (stat.st_mtime, stat.st_size, meta)
    return meta
