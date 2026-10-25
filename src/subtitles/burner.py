"""Burn subtitles into MP4 video clips using local FFmpeg."""

from pathlib import Path
import subprocess
from typing import Optional


class SubtitleBurnError(RuntimeError):
    """Raised when burning subtitles into video fails."""
    pass


def escape_ffmpeg_filter_path(file_path: Path) -> str:
    """
    Escape a filesystem path for inclusion in FFmpeg filter strings.
    Converts backslashes, escapes colons (crucial on Windows and absolute Unix paths),
    and single quotes.
    """
    posix_str = str(file_path.resolve()).replace("\\", "/")
    # Escape colons, backslashes, and single quotes for ffmpeg filtergraph
    escaped = posix_str.replace(":", "\\:").replace("'", "\\'")
    return escaped


def burn_subtitles_to_video(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    font_size: int = 18,
    margin_v: int = 24,
    timeout_secs: int = 120,
) -> Path:
    """
    Burn standard SubRip (.srt) subtitles directly into an MP4 clip using local FFmpeg.

    Keeps the input clip completely unmodified.
    Renders high-contrast, readable subtitles with dark outline.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video file not found: {video_path}")
    if not srt_path.is_file():
        raise FileNotFoundError(f"SRT subtitle file not found: {srt_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Style string for libass/subtitles filter
    # PrimaryColour=&H00FFFFFF (opaque white), OutlineColour=&H00000000 (black outline)
    style = (
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=3,"
        f"Outline=1.5,"
        f"Shadow=0,"
        f"MarginV={margin_v}"
    )

    escaped_srt = escape_ffmpeg_filter_path(srt_path)
    vf_filter = f"subtitles='{escaped_srt}':force_style='{style}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_secs,
        )
        if res.returncode != 0:
            raise SubtitleBurnError(
                f"FFmpeg failed to burn subtitles into video (code {res.returncode}): {res.stderr.strip()}"
            )

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise SubtitleBurnError(f"Subtitled output file was not created or is empty: {output_path}")

        return output_path

    except subprocess.TimeoutExpired as exc:
        raise SubtitleBurnError(f"FFmpeg subtitle burn-in timed out after {timeout_secs} seconds.") from exc
    except Exception as exc:
        if isinstance(exc, SubtitleBurnError):
            raise
        raise SubtitleBurnError(f"Unexpected error burning subtitles: {exc}") from exc
