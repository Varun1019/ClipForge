"""FFmpeg-based reframing engine for vertical (9:16) and square (1:1) video exports."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from typing import Optional, Tuple

from src.framing.options import AspectRatioMode, FramingConfig, FramingStyle
from src.subtitles.burner import escape_ffmpeg_filter_path

logger = logging.getLogger("clipforge.framing")


class FramingError(RuntimeError):
    """Raised when reframing video fails."""
    pass


def build_reframe_filter(
    config: FramingConfig,
    srt_path: Optional[Path] = None,
    font_size: int = 20,
    margin_v: int = 40,
) -> Tuple[list[str], bool]:
    """
    Build FFmpeg filter command-line arguments for reframing.

    Returns:
        tuple (filter_args: list[str], is_filter_complex: bool)
    """
    config.validate()
    w = config.target_width
    h = config.target_height

    subtitle_filter = ""
    if srt_path is not None and srt_path.is_file():
        escaped_srt = escape_ffmpeg_filter_path(srt_path)
        style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"BorderStyle=3,"
            f"Outline=2,"
            f"Alignment=2,"
            f"MarginV={margin_v}"
        )
        subtitle_filter = f",subtitles='{escaped_srt}':force_style='{style}'"

    if config.style == FramingStyle.CENTER_CROP:
        # Scale to fill canvas while preserving aspect ratio, then crop center
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2{subtitle_filter}"
        )
        return (["-vf", vf], False)

    elif config.style == FramingStyle.BLURRED_BACKGROUND:
        # Optimized 2-pass blur: downscale ambient background 4x before blurring, then scale to target canvas
        bg_w = max(2, (w // 4) // 2 * 2)
        bg_h = max(2, (h // 4) // 2 * 2)
        filter_complex = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,crop={bg_w}:{bg_h},"
            f"boxblur=8:2,scale={w}:{h}:flags=bilinear[bg_blur];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2[fg_fit];"
            f"[bg_blur][fg_fit]overlay=(W-w)/2:(H-h)/2{subtitle_filter}[outv]"
        )
        return (["-filter_complex", filter_complex, "-map", "[outv]"], True)

    else:
        raise ValueError(f"Unsupported framing style: {config.style}")


def reframe_video(
    input_path: Path,
    output_path: Path,
    config: FramingConfig,
    srt_path: Optional[Path] = None,
    timeout_secs: int = 180,
) -> Path:
    """
    Reframe an input video to the specified aspect ratio and canvas using local FFmpeg.

    Preserves input intact. Preserves audio synchronization.
    Supports optional burned-in subtitles positioned cleanly on the reframed canvas.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Source video file not found: {input_path}")

    config.validate()
    if not config.is_reframe_needed():
        return input_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_args, is_complex = build_reframe_filter(config, srt_path=srt_path)

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    cmd.extend(filter_args)
    if not is_complex:
        cmd.extend(["-map", "0:v:0"])
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-map", "0:a?",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info("Executing reframe command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_secs,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FramingError(
            f"FFmpeg reframing timed out after {timeout_secs} seconds for {input_path.name}"
        ) from exc
    except Exception as exc:
        raise FramingError(f"Failed to execute FFmpeg for reframing: {exc}") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or proc.stdout.strip() or "Unknown FFmpeg error"
        relevant_err = "\n".join(err_msg.splitlines()[-10:])
        raise FramingError(f"FFmpeg reframing failed (code {proc.returncode}):\n{relevant_err}")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise FramingError(f"Reframing completed but output file is missing or empty: {output_path}")

    logger.info("Reframed video saved to %s", output_path)
    return output_path
