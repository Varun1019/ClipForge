"""Provider registry for ClipForge AI integrations."""

from typing import Optional
from src.providers.base import BaseAIProvider
from src.providers.gemini_provider import GoogleGeminiProvider


class ProviderRegistry:
    """Manages available and planned AI providers."""

    def __init__(self):
        self._providers: dict[str, BaseAIProvider] = {}
        # Register default initial provider
        self.register_provider(GoogleGeminiProvider())

    def register_provider(self, provider: BaseAIProvider) -> None:
        """Register a new provider instance."""
        self._providers[provider.provider_id] = provider

    def get_provider(self, provider_id: str) -> Optional[BaseAIProvider]:
        """Retrieve a provider by its unique ID."""
        return self._providers.get(provider_id)

    def list_providers(self) -> list[dict]:
        """List all available registered providers and their metadata."""
        return [
            {
                "id": p.provider_id,
                "name": p.display_name,
                "default_model": p.default_model,
                "supported_models": p.supported_models,
                "status": "ready" if p.provider_id == "gemini" else "planned",
            }
            for p in self._providers.values()
        ]


# Global singleton instance
provider_registry = ProviderRegistry()
