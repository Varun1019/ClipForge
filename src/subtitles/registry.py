"""Registry for modular speech-to-text transcription providers."""

from typing import Optional
from src.subtitles.base import BaseTranscriptionProvider
from src.subtitles.whisper_provider import FasterWhisperProvider


class TranscriptionRegistry:
    """Manages available transcription engine providers."""

    def __init__(self):
        self._providers: dict[str, BaseTranscriptionProvider] = {}
        # Register default local Faster-Whisper provider
        self.register_provider(FasterWhisperProvider())

    def register_provider(self, provider: BaseTranscriptionProvider) -> None:
        """Register a new transcription engine."""
        self._providers[provider.provider_id] = provider

    def get_provider(self, provider_id: str = "faster-whisper") -> Optional[BaseTranscriptionProvider]:
        """Retrieve transcription provider by ID."""
        return self._providers.get(provider_id)

    def list_providers(self) -> list[dict]:
        """Return metadata for all registered transcription providers."""
        return [
            {
                "id": p.provider_id,
                "name": p.display_name,
                "available": p.is_available(),
            }
            for p in self._providers.values()
        ]


# Singleton instance
transcription_registry = TranscriptionRegistry()
