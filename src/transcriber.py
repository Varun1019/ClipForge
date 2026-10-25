"""ClipForge local speech-to-text transcription engine using Faster-Whisper."""

import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Optional

import ctranslate2
from faster_whisper import WhisperModel


class TranscriptionError(Exception):
    """Base exception for all transcription pipeline failures."""
    pass


class NoAudioStreamError(TranscriptionError):
    """Raised when the target media file contains no audio stream."""
    pass


class AudioExtractionError(TranscriptionError):
    """Raised when extracting audio via FFmpeg fails."""
    pass


class ModelExecutionError(TranscriptionError):
    """Raised when the Whisper model fails to load or execute."""
    pass


def check_cuda_runtime_available() -> tuple[bool, str]:
    """
    Check if NVIDIA GPU and cuBLAS/CUDA runtime libraries are truly usable.
    Returns (is_available, reason_description).
    """
    if ctranslate2.get_cuda_device_count() == 0:
        return False, "No NVIDIA GPU detected."

    # Look for libcublas.so.12 which CTranslate2 requires on Linux for CUDA
    cublas = ctypes.util.find_library("cublas") or ctypes.util.find_library("cublas.so.12")
    if not cublas:
        # Check standard Linux library locations directly
        lib_paths = (
            "/usr/lib/libcublas.so.12",
            "/usr/lib64/libcublas.so.12",
            "/usr/local/cuda/lib64/libcublas.so.12",
        )
        if not any(Path(p).is_file() for p in lib_paths):
            return (
                False,
                "NVIDIA GPU detected, but CUDA runtime library (libcublas.so.12) is not installed.",
            )
    return True, "NVIDIA GPU and CUDA runtime verified."


def detect_compute_device() -> tuple[str, str, str]:
    """
    Resolve device and compute type. Defaults reliably to CPU (int8).

    Returns:
        (device, compute_type, description)
    """
    cuda_ok, cuda_reason = check_cuda_runtime_available()
    if cuda_ok:
        return "cuda", "float16", "NVIDIA RTX 3050 GPU (CUDA float16)"

    # Reliable default: CPU with int8 quantization (fast on AMD Ryzen 7)
    return "cpu", "int8", f"CPU (int8 quantized) - {cuda_reason}"


def check_audio_stream_exists(media_path: Path | str) -> bool:
    """Check if the target video/media file contains at least one audio stream using ffprobe."""
    path = Path(media_path).resolve()
    if not path.is_file():
        return False

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index,codec_name",
        "-of", "json",
        str(path),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode != 0:
            return False
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        return len(streams) > 0
    except Exception:
        return False


def compute_file_hash(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file efficiently in chunks."""
    sha = hashlib.sha256()
    path = Path(file_path)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into standard SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def generate_srt(segments: list[dict]) -> str:
    """Convert segment list into standard SubRip (.srt) subtitle string."""
    srt_lines = []
    for idx, seg in enumerate(segments, 1):
        start_ts = format_srt_timestamp(seg["start"])
        end_ts = format_srt_timestamp(seg["end"])
        srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{seg['text']}\n")
    return "\n".join(srt_lines)


def get_cache_path(
    video_hash: str,
    model_size: str,
    cache_dir: Path = Path("temp/transcripts"),
) -> Path:
    """Return JSON cache path for transcript."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"transcript_{video_hash}_{model_size}.json"


def load_cached_transcript(
    video_hash: str,
    model_size: str,
    cache_dir: Path = Path("temp/transcripts"),
) -> Optional[dict]:
    """Load valid cached transcript if available."""
    cache_file = get_cache_path(video_hash, model_size, cache_dir)
    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "segments" in data:
                    data["from_cache"] = True
                    return data
        except Exception:
            return None
    return None


def save_transcript_to_cache(
    transcript_data: dict,
    video_hash: str,
    model_size: str,
    cache_dir: Path = Path("temp/transcripts"),
) -> Path:
    """Save structured transcript data to JSON cache."""
    cache_file = get_cache_path(video_hash, model_size, cache_dir)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    return cache_file


def extract_audio_from_video(
    video_path: Path | str,
    output_wav_path: Optional[Path | str] = None,
) -> Path:
    """
    Extract 16kHz mono PCM WAV audio from video using FFmpeg.

    Raises:
        FileNotFoundError: If source video does not exist.
        NoAudioStreamError: If source video has no audio track.
        AudioExtractionError: If FFmpeg extraction fails.
    """
    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")

    if not check_audio_stream_exists(source):
        raise NoAudioStreamError(
            f"The video '{source.name}' has no audio stream. "
            "Speech transcription requires an audio track."
        )

    if not shutil.which("ffmpeg"):
        raise AudioExtractionError("FFmpeg executable not found in system PATH.")

    if output_wav_path is None:
        dest = source.parent / f"{source.stem}_audio_16k.wav"
    else:
        dest = Path(output_wav_path).resolve()

    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(dest),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExtractionError("FFmpeg audio extraction timed out.") from exc
    except Exception as exc:
        raise AudioExtractionError(f"Error launching FFmpeg: {exc}") from exc

    if res.returncode != 0:
        raise AudioExtractionError(f"FFmpeg audio extraction failed: {res.stderr.strip()}")

    if not dest.is_file() or dest.stat().st_size == 0:
        raise AudioExtractionError("Extracted audio file is empty or corrupted.")

    return dest


def transcribe_video(
    video_path: Path | str,
    model_size: str = "tiny",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    force_refresh: bool = False,
    cache_dir: Optional[Path] = None,
    download_root: Path = Path("temp/models"),
    vad_filter: bool = False,
) -> dict:
    """
    Transcribe speech in video file using Faster-Whisper locally.

    Returns structured transcript dictionary with segment timestamps and words.

    Raises:
        FileNotFoundError: If video file is missing.
        NoAudioStreamError: If video contains no audio stream.
        AudioExtractionError: If audio extraction fails.
        ModelExecutionError: If Faster-Whisper fails during inference.
    """
    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")

    # Verify audio stream exists
    if not check_audio_stream_exists(source):
        raise NoAudioStreamError(
            f"The video '{source.name}' has no audio track. "
            "Speech transcription requires an audio stream."
        )

    video_hash = compute_file_hash(source)
    target_cache_dir = cache_dir or Path("temp/transcripts")

    # Check cache first
    if not force_refresh:
        cached = load_cached_transcript(video_hash, model_size, cache_dir=target_cache_dir)
        if cached is not None:
            return cached

    # Resolve hardware execution device (reliable CPU default)
    auto_dev, auto_comp, _ = detect_compute_device()
    target_device = device or auto_dev
    target_compute = compute_type or auto_comp

    # Step 1: Extract 16kHz mono WAV using FFmpeg
    temp_wav = Path("temp") / f"extract_{video_hash}.wav"
    extract_audio_from_video(source, temp_wav)

    try:
        # Step 2: Initialize Faster-Whisper model
        download_root.mkdir(parents=True, exist_ok=True)
        try:
            model = WhisperModel(
                model_size_or_path=model_size,
                device=target_device,
                compute_type=target_compute,
                download_root=str(download_root),
            )
        except Exception as exc:
            # If target_device was CUDA and failed, retry once on CPU
            if target_device != "cpu":
                try:
                    target_device = "cpu"
                    target_compute = "int8"
                    model = WhisperModel(
                        model_size_or_path=model_size,
                        device=target_device,
                        compute_type=target_compute,
                        download_root=str(download_root),
                    )
                except Exception as exc2:
                    raise ModelExecutionError(f"Failed to load Whisper model on CPU: {exc2}") from exc2
            else:
                raise ModelExecutionError(f"Failed to load Whisper model: {exc}") from exc

        # Step 3: Run local transcription with word-level timestamps
        try:
            segments_gen, info = model.transcribe(
                str(temp_wav),
                beam_size=1,
                word_timestamps=True,
                vad_filter=vad_filter,
            )
            raw_segments = list(segments_gen)
        except Exception as exc:
            raise ModelExecutionError(f"Whisper inference error: {exc}") from exc

        segments_list = []
        full_text_parts = []

        for seg in raw_segments:
            clean_text = seg.text.strip()
            if clean_text:
                full_text_parts.append(clean_text)

            words_data = []
            if seg.words:
                for w in seg.words:
                    words_data.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 4),
                    })

            segments_list.append({
                "id": seg.id,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": clean_text,
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "words": words_data,
            })

        has_speech = len(segments_list) > 0
        transcript_data = {
            "source_file": source.name,
            "video_hash": video_hash,
            "model_size": model_size,
            "device_used": target_device,
            "compute_type": target_compute,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
            "segments_count": len(segments_list),
            "status": "success" if has_speech else "no_speech_detected",
            "full_text": " ".join(full_text_parts),
            "segments": segments_list,
            "created_at": time.time(),
            "from_cache": False,
        }

        # Step 4: Cache results only if speech was detected or if explicit
        save_transcript_to_cache(transcript_data, video_hash, model_size, cache_dir=target_cache_dir)
        return transcript_data

    finally:
        # Step 5: Clean up temporary audio file
        if temp_wav.exists():
            try:
                os.remove(temp_wav)
            except OSError:
                pass
