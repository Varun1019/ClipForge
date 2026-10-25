"""Local speech-to-text provider using Faster-Whisper and CTranslate2 with GPU/CPU auto-detection."""

from pathlib import Path
from typing import Callable, Optional

from src.subtitles.base import (
    BaseTranscriptionProvider,
    SubtitleSegment,
    TranscriptionResult,
    WordTimestamp,
)


def _log_stage(msg: str, callback: Optional[Callable[[str], None]] = None) -> None:
    """Print flushed progress message and trigger UI callback if provided."""
    print(f"[ClipForge-Subtitles] {msg}", flush=True)
    if callback:
        try:
            callback(msg)
        except Exception:
            pass


_CACHED_DEVICE_INFO: Optional[tuple[str, str]] = None


class FasterWhisperProvider(BaseTranscriptionProvider):
    """
    Local transcription engine powered by Faster-Whisper (CTranslate2).
    Accelerated with NVIDIA CUDA on laptop/desktop GPUs, with seamless CPU int8 fallback.
    No PyTorch dependency required.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        models_dir: Optional[Path] = None,
    ):
        self.model_size = model_size
        self.models_dir = models_dir or Path("temp/models")
        self._model = None
        self._device = None
        self._compute_type = None

    @property
    def provider_id(self) -> str:
        return "faster-whisper"

    @property
    def display_name(self) -> str:
        return f"Faster-Whisper ({self.model_size})"

    def is_available(self) -> bool:
        """Verify that faster_whisper and ctranslate2 are importable."""
        try:
            import ctranslate2  # noqa: F401
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _determine_device(self) -> tuple[str, str]:
        """Detect whether CUDA GPU is usable (with required libraries) or fall back to CPU."""
        global _CACHED_DEVICE_INFO
        if _CACHED_DEVICE_INFO is not None:
            return _CACHED_DEVICE_INFO

        device, compute_type = "cpu", "int8"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                import ctypes
                try:
                    ctypes.CDLL("libcublas.so.12")
                    device, compute_type = "cuda", "float16"
                except Exception:
                    pass
        except Exception:
            pass

        _CACHED_DEVICE_INFO = (device, compute_type)
        return _CACHED_DEVICE_INFO

    def _get_model(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Lazy-load the Whisper model only when transcription is triggered."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        self.models_dir.mkdir(parents=True, exist_ok=True)
        device, compute_type = self._determine_device()

        _log_stage(f"Loading Faster-Whisper '{self.model_size}' model on {device.upper()} ({compute_type})...", progress_callback)
        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(self.models_dir),
            )
            self._device = device
            self._compute_type = compute_type
            _log_stage(f"Whisper model loaded successfully on {device.upper()}.", progress_callback)
            return self._model
        except Exception as exc:
            if device == "cuda":
                _log_stage(f"CUDA initialization failed ({exc}). Retrying on CPU (int8)...", progress_callback)
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.models_dir),
                )
                self._device = "cpu"
                self._compute_type = "int8"
                _log_stage("Whisper model loaded on CPU fallback.", progress_callback)
                return self._model
            raise

    def transcribe(
        self,
        audio_or_video_path: Path,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe speech from a media file.
        Returns TranscriptionResult containing segments and word-level timestamps.
        """
        if not audio_or_video_path.is_file():
            raise FileNotFoundError(f"Media file to transcribe not found: {audio_or_video_path}")

        model = self._get_model(progress_callback)
        _log_stage(f"Transcribing '{audio_or_video_path.name}'...", progress_callback)

        try:
            segments_iter, info = model.transcribe(
                str(audio_or_video_path),
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=False,  # Avoid aggressive filtering of soft vocals/dialogue
                beam_size=2,
            )
            raw_segments = list(segments_iter)
        except (IndexError, ValueError):
            # PyAV raises IndexError / ValueError when file lacks an audio stream
            _log_stage(f"No audio stream detected in '{audio_or_video_path.name}'.", progress_callback)
            return TranscriptionResult(
                segments=[],
                language=language or "en",
                duration=0.0,
            )
        except RuntimeError as exc:
            err_str = str(exc).lower()
            if "cuda" in err_str or "cublas" in err_str or "cudnn" in err_str:
                _log_stage(f"CUDA execution failed ({exc}). Retrying on CPU (int8)...", progress_callback)
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.models_dir),
                )
                self._device = "cpu"
                self._compute_type = "int8"
                segments_iter, info = self._model.transcribe(
                    str(audio_or_video_path),
                    language=language,
                    word_timestamps=word_timestamps,
                    vad_filter=False,
                    beam_size=2,
                )
                raw_segments = list(segments_iter)
            else:
                raise

        detected_language = getattr(info, "language", language or "en")
        parsed_segments: list[SubtitleSegment] = []

        for seg_idx, seg in enumerate(raw_segments, 1):
            words: list[WordTimestamp] = []
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append(
                        WordTimestamp(
                            word=w.word,
                            start=float(w.start),
                            end=float(w.end),
                            probability=float(getattr(w, "probability", 1.0)),
                        )
                    )

            parsed_segments.append(
                SubtitleSegment(
                    id=seg_idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )

        duration = getattr(info, "duration", 0.0)
        _log_stage(
            f"Transcription finished: {len(parsed_segments)} segments detected "
            f"(Language: {detected_language.upper()}).",
            progress_callback,
        )

        return TranscriptionResult(
            segments=parsed_segments,
            language=detected_language,
            duration=duration,
        )
