"""Base interfaces and data structures for ClipForge AI highlight providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class InputModality(str, Enum):
    """Input strategy supported by the model for analyzing media."""
    NATIVE_VIDEO = "native_video"        # Video file uploaded directly (e.g. Gemini 2.5)
    AUDIO_EXTRACTED = "audio_extracted"  # Audio extracted via FFmpeg (efficient token usage)
    TRANSCRIPT_TEXT = "transcript_text"  # Transcribed text with timestamps (text-only LLMs)


class ModelCapability(str, Enum):
    """Specific capabilities offered by an AI model."""
    VIDEO_UNDERSTANDING = "video_understanding"
    AUDIO_UNDERSTANDING = "audio_understanding"
    STRUCTURED_OUTPUT = "structured_output"
    LARGE_CONTEXT_WINDOW = "large_context_window"  # 1M+ tokens (e.g. Gemini 2.5/2.0)
    LOW_LATENCY = "low_latency"


class ProviderError(Exception):
    """Base exception for provider operations."""
    pass


class InvalidAPIKeyError(ProviderError):
    """Raised when the provided API key is rejected by the provider."""
    pass


class QuotaExceededError(ProviderError):
    """Raised when rate limit or billing quota is exhausted."""
    pass


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is unavailable or unrecognized."""
    pass


class InputSizeExceededError(ProviderError):
    """Raised when the media payload exceeds the model or upload limits."""
    pass


@dataclass
class HighlightCandidate:
    """A single proposed highlight clip identified by AI."""
    title: str
    start_time: float
    end_time: float
    duration: float
    explanation: str
    hook: str
    confidence_score: Optional[float] = None

    def validate(self, max_duration: float) -> tuple[bool, Optional[str]]:
        """Validate candidate timestamps against source video limits."""
        if self.start_time < 0.0:
            return False, f"Start time ({self.start_time:.2f}s) cannot be negative."
        if self.end_time <= self.start_time:
            return False, f"End time ({self.end_time:.2f}s) must be after start time ({self.start_time:.2f}s)."
        if (self.end_time - self.start_time) < 5.0:
            return False, f"Clip duration ({self.end_time - self.start_time:.2f}s) is too short. Minimum duration is 5.0s."
        if self.end_time > (max_duration + 0.5):
            return False, f"End time ({self.end_time:.2f}s) exceeds video duration ({max_duration:.2f}s)."
        return True, None


@dataclass
class HighlightRequest:
    """User-configured parameters for AI video analysis."""
    video_path: Path
    video_duration: float
    file_size_bytes: int
    target_clip_count: int = 3
    target_clip_duration_secs: float = 45.0
    custom_instructions: Optional[str] = None
    preferred_modality: Optional[InputModality] = None


@dataclass
class HighlightResponse:
    """Standardized response from an AI highlight provider."""
    provider_id: str
    model_name: str
    highlights: list[HighlightCandidate]
    input_modality_used: InputModality
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[str] = None


class BaseAIProvider(ABC):
    """
    Abstract interface for AI providers in ClipForge.

    Enables plug-and-play addition of cloud (Gemini, OpenAI, Anthropic)
    or local providers without altering the clipping pipeline.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique machine identifier for the provider (e.g. 'gemini')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name (e.g. 'Google Gemini')."""
        pass

    def list_available_models(
        self,
        api_key: str,
        timeout_secs: float = 10.0,
    ) -> list[str]:
        """
        Dynamically query and return available models for the given credentials.
        Default implementation returns static supported_models.
        """
        return self.supported_models

    def choose_default_model(self, available_models: list[str]) -> str:
        """Pick a sensible, cost-effective default model from the available models."""
        if available_models:
            return available_models[0]
        return self.default_model

    def verify_model_access(
        self,
        api_key: str,
        model_name: str,
        timeout_secs: float = 10.0,
    ) -> tuple[bool, str]:
        """Verify that a selected model is usable before running operations."""
        return True, f"Model '{model_name}' is accessible."

    @abstractmethod
    def validate_api_key_format(self, api_key: str) -> tuple[bool, str]:
        """Perform basic syntactic validation before making network requests."""
        pass

    def verify_api_key(
        self,
        api_key: str,
        check_live: bool = True,
        timeout_secs: float = 10.0,
    ) -> tuple[bool, str]:
        """
        Verify the API key against syntactic rules and optional live service authentication.
        Default implementation delegates to validate_api_key_format.
        """
        return self.validate_api_key_format(api_key)

    @abstractmethod
    def get_model_capabilities(self, model_name: str) -> set[ModelCapability]:
        """Return the verified capabilities for a specific model."""
        pass

    @abstractmethod
    def resolve_input_strategy(
        self,
        model_name: str,
        file_size_bytes: int,
        duration_secs: float,
    ) -> InputModality:
        """
        Determine optimal data representation (audio vs video) based on model limits.
        Extracting audio via FFmpeg is often 10x-30x smaller than raw video.
        """
        pass

    @abstractmethod
    def analyze_highlights(
        self,
        request: HighlightRequest,
        api_key: str,
        model_name: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> HighlightResponse:
        """
        Execute AI analysis using the user's API key and return structured moments.
        """
        pass
