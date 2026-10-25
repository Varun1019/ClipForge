"""ClipForge video clipping engine using FFmpeg."""

from pathlib import Path
import math
import shutil
import subprocess
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from src.framing.options import FramingConfig


class VideoClipperError(Exception):
    """Raised when video clipping fails during FFmpeg execution or filesystem operations."""
    pass


class TimestampValidationError(ValueError):
    """Raised when provided start and end timestamps violate clipping constraints."""
    pass


def validate_timestamps(start_time: float, end_time: float, max_duration: float) -> None:
    """
    Validate clipping timestamps against source constraints.

    Rules:
    - Start and end time must be finite numbers (not NaN or Inf).
    - Start time must be 0.0 or greater.
    - End time must be strictly greater than start time.
    - End time cannot exceed total source duration.
    """
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise TimestampValidationError("Timestamps must be finite numbers.")

    if start_time < 0.0:
        raise TimestampValidationError(
            f"Start time ({start_time:.2f}s) cannot be negative. Must be 0.0 or greater."
        )

    if end_time <= start_time:
        raise TimestampValidationError(
            f"End time ({end_time:.2f}s) must be greater than start time ({start_time:.2f}s)."
        )

    # Allow a tiny epsilon (0.05s) to avoid strict float precision rejections at the boundary
    if end_time > (max_duration + 0.05):
        raise TimestampValidationError(
            f"End time ({end_time:.2f}s) exceeds source video duration ({max_duration:.2f}s)."
        )


def build_ffmpeg_clip_command(
    input_path: Path | str,
    output_path: Path | str,
    start_time: float,
    end_time: float,
) -> list[str]:
    """
    Construct safe argument list for FFmpeg sub-clipping without shell=True.

    Uses frame-accurate seek with standard H.264 / AAC web-compatible encoding.
    """
    inp = str(Path(input_path).resolve())
    out = str(Path(output_path).resolve())

    # Format timestamps cleanly with up to 3 decimal places
    start_str = f"{start_time:.3f}".rstrip("0").rstrip(".") if "." in f"{start_time:.3f}" else str(start_time)
    end_str = f"{end_time:.3f}".rstrip("0").rstrip(".") if "." in f"{end_time:.3f}" else str(end_time)

    return [
        "ffmpeg",
        "-y",
        "-ss", start_str,
        "-to", end_str,
        "-i", inp,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        out,
    ]


def create_video_clip(
    input_path: Path | str,
    output_path: Path | str,
    start_time: float,
    end_time: float,
    max_duration: Optional[float] = None,
) -> Path:
    """
    Clip an interval from a source video into an MP4 file using FFmpeg.

    Args:
        input_path: Path to source video file.
        output_path: Destination path for created clip.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        max_duration: Optional total source video duration for validation.

    Returns:
        Path object pointing to the created clip.

    Raises:
        TimestampValidationError: When timestamp range is invalid.
        VideoClipperError: When FFmpeg fails or cannot be executed.
        FileNotFoundError: When source video does not exist.
    """
    source = Path(input_path).resolve()
    dest = Path(output_path).resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Source video file not found: {source}")

    if source == dest:
        raise VideoClipperError("Destination path cannot be identical to source video path.")

    if max_duration is not None:
        validate_timestamps(start_time, end_time, max_duration)
    elif start_time < 0.0 or end_time <= start_time:
        raise TimestampValidationError("Start time must be >= 0 and end time must be > start time.")

    if not shutil.which("ffmpeg"):
        raise VideoClipperError("FFmpeg executable not found in system PATH.")

    # Ensure output destination directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_ffmpeg_clip_command(source, dest, start_time, end_time)

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoClipperError("FFmpeg clipping process timed out.") from exc
    except Exception as exc:
        raise VideoClipperError(f"Subprocess execution error: {exc}") from exc

    if result.returncode != 0:
        err_details = result.stderr.strip() or "Unknown FFmpeg error"
        raise VideoClipperError(f"FFmpeg clipping failed (exit {result.returncode}): {err_details}")

    if not dest.is_file() or dest.stat().st_size == 0:
        raise VideoClipperError("FFmpeg finished but output clip file was empty or missing.")

    return dest


def batch_create_clips(
    source_path: Path | str,
    candidates: list,
    output_dir: Path | str = Path("exports"),
    max_duration: Optional[float] = None,
    generate_subtitles: bool = False,
    burn_subtitles: bool = False,
    transcription_provider=None,
    framing_config: Optional["FramingConfig"] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """
    Generate multiple MP4 clips from validated candidates using FFmpeg.
    Optionally transcribes dialogue, generates an SRT subtitle file, burns in captions,
    and reframes clips for vertical (9:16) or square (1:1) export.

    Args:
        source_path: Path to the source video.
        candidates: List of HighlightCandidate objects or dicts with start_time, end_time, title.
        output_dir: Destination directory for created clips.
        max_duration: Optional video duration limit.
        generate_subtitles: If True, generate an SRT file for each created clip.
        burn_subtitles: If True, render subtitles directly into a separate copy of the MP4.
        transcription_provider: Optional custom BaseTranscriptionProvider instance.
        framing_config: Optional FramingConfig specifying aspect ratio and formatting style.
        progress_callback: Optional callback for UI progress reporting.

    Returns:
        List of dicts containing:
        - 'title': Clip title
        - 'start_time': Start time in seconds
        - 'end_time': End time in seconds
        - 'duration': Duration in seconds
        - 'output_path': Path to clean generated MP4 (or None if failed)
        - 'srt_path': Path to generated .srt file (if subtitles enabled and speech found)
        - 'subtitled_path': Path to burned-in MP4 (if burn_subtitles enabled)
        - 'reframed_path': Path to reframed MP4 (if framing_config enabled)
        - 'reframed_subtitled_path': Path to reframed subtitled MP4 (if both enabled)
        - 'has_speech': True if speech was transcribed
        - 'transcript': Clean transcribed dialogue text
        - 'success': True if clip was created successfully
        - 'error': Error message if failed
    """
    import re
    import time

    source = Path(source_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    base_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", source.stem)

    for idx, cand in enumerate(candidates, 1):
        # Extract attributes from candidate object or dict
        if hasattr(cand, "start_time"):
            start = cand.start_time
            end = cand.end_time
            title = cand.title
        else:
            start = cand.get("start_time", 0.0)
            end = cand.get("end_time", 0.0)
            title = cand.get("title", f"Clip_{idx}")

        clip_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", title).strip("_") or f"clip_{idx}"
        timestamp_tag = int(time.time())
        dest_filename = f"{base_stem}_{idx:02d}_{clip_stem}_{start:.1f}s_to_{end:.1f}s_{timestamp_tag}.mp4"
        dest_path = out_dir / dest_filename

        try:
            if progress_callback:
                progress_callback(f"Rendering clip {idx}/{len(candidates)}: '{title}'...")

            created = create_video_clip(
                input_path=source,
                output_path=dest_path,
                start_time=start,
                end_time=end,
                max_duration=max_duration,
            )

            srt_path = None
            subtitled_path = None
            reframed_path = None
            reframed_subtitled_path = None
            has_speech = False
            transcript_text = ""

            if generate_subtitles:
                try:
                    if progress_callback:
                        progress_callback(f"Transcribing audio for clip {idx}: '{title}'...")

                    from src.subtitles import burn_subtitles_to_video, transcription_registry
                    transcriber = transcription_provider or transcription_registry.get_provider()

                    if transcriber and transcriber.is_available():
                        trans_res = transcriber.transcribe(
                            created,
                            progress_callback=progress_callback,
                        )
                        has_speech = trans_res.has_speech
                        if has_speech:
                            srt_path = dest_path.with_suffix(".srt")
                            # Subtitle timings for exported clip begin at 0.0s
                            srt_content = trans_res.to_srt(offset_secs=0.0)
                            srt_path.write_text(srt_content, encoding="utf-8")
                            transcript_text = "\n".join(s.text for s in trans_res.segments)

                            if burn_subtitles:
                                if progress_callback:
                                    progress_callback(f"Burning subtitles into clip {idx}: '{title}'...")
                                subtitled_path = dest_path.with_name(f"{dest_path.stem}_subtitled.mp4")
                                burn_subtitles_to_video(
                                    video_path=created,
                                    srt_path=srt_path,
                                    output_path=subtitled_path,
                                )
                        else:
                            if progress_callback:
                                progress_callback(f"No speech detected in clip {idx}: '{title}'.")
                except Exception as sub_exc:
                    if progress_callback:
                        progress_callback(f"Subtitle notice for clip {idx}: {sub_exc}")

            if framing_config and framing_config.is_reframe_needed():
                try:
                    from src.framing import reframe_video
                    suffix = framing_config.get_suffix()
                    reframed_path = dest_path.with_name(f"{dest_path.stem}{suffix}.mp4")
                    if progress_callback:
                        progress_callback(f"Reframing clip {idx} ({framing_config.aspect_ratio.value})...")
                    reframe_video(
                        input_path=created,
                        output_path=reframed_path,
                        config=framing_config,
                    )

                    if burn_subtitles and srt_path and srt_path.is_file():
                        reframed_subtitled_path = dest_path.with_name(f"{dest_path.stem}{suffix}_subtitled.mp4")
                        if progress_callback:
                            progress_callback(f"Rendering subtitles on reframed clip {idx}...")
                        reframe_video(
                            input_path=created,
                            output_path=reframed_subtitled_path,
                            config=framing_config,
                            srt_path=srt_path,
                        )
                except Exception as frame_exc:
                    if progress_callback:
                        progress_callback(f"Reframing notice for clip {idx}: {frame_exc}")

            results.append({
                "title": title,
                "start_time": start,
                "end_time": end,
                "duration": round(end - start, 2),
                "output_path": created,
                "srt_path": srt_path,
                "subtitled_path": subtitled_path,
                "reframed_path": reframed_path,
                "reframed_subtitled_path": reframed_subtitled_path,
                "has_speech": has_speech,
                "transcript": transcript_text,
                "success": True,
                "error": None,
            })
        except Exception as exc:
            results.append({
                "title": title,
                "start_time": start,
                "end_time": end,
                "duration": round(end - start, 2) if end > start else 0.0,
                "output_path": None,
                "srt_path": None,
                "subtitled_path": None,
                "reframed_path": None,
                "reframed_subtitled_path": None,
                "has_speech": False,
                "transcript": "",
                "success": False,
                "error": str(exc),
            })

    return results

