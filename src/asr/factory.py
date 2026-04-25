"""Decorator-based ASR provider registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .interface import ASRInterface


@dataclass(frozen=True)
class ASRProviderMetadata:
    """Static metadata shown in provider list responses."""

    name: str
    display_name: str
    provider_type: str
    supports_backend_transcription: bool
    supports_browser_streaming: bool
    description: str


class ASRFactory:
    """Class-scoped registry mapping provider name to provider class."""

    _registry: dict[str, type[ASRInterface]] = {}
    _metadata: dict[str, ASRProviderMetadata] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        metadata: ASRProviderMetadata,
    ) -> Callable[[type[ASRInterface]], type[ASRInterface]]:
        """Return a decorator that registers a provider class."""

        def wrapper(provider_class: type[ASRInterface]) -> type[ASRInterface]:
            provider_class.provider_name = name
            provider_class.supports_backend_transcription = metadata.supports_backend_transcription
            provider_class.supports_browser_streaming = metadata.supports_browser_streaming
            cls._registry[name] = provider_class
            cls._metadata[name] = metadata
            return provider_class

        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ASRInterface:
        """Instantiate a registered provider by name."""

        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(f"Unknown ASR provider: {name!r}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return sorted registered provider names."""

        return sorted(cls._registry.keys())

    @classmethod
    def metadata(cls, name: str) -> ASRProviderMetadata:
        """Return static metadata for a registered provider."""

        if name not in cls._metadata:
            available = sorted(cls._metadata.keys())
            raise ValueError(f"Unknown ASR provider metadata: {name!r}. Available: {available}")
        return cls._metadata[name]
