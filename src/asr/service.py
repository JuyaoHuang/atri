"""ASR application service."""

from __future__ import annotations

from typing import Any

from . import providers as _providers  # noqa: F401
from .config import ASRConfigStore
from .exceptions import ASRConfigError, ASRProviderUnavailableError
from .factory import ASRFactory

SENSITIVE_CONFIG_KEYS = {"api_key", "token", "secret", "password"}
SENSITIVE_CONFIG_MASK = "********"


class ASRService:
    """Coordinate ASR config, provider health, switching, and transcription."""

    def __init__(self, config_store: ASRConfigStore) -> None:
        self.config_store = config_store

    def get_config(self) -> dict[str, Any]:
        """Return persisted OLV-shaped ASR config."""

        return self._public_config(self.config_store.read())

    def update_config(self, patch: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
        """Merge and persist a partial OLV-shaped ASR config update."""

        patch = self._strip_masked_sensitive_values(patch)
        next_model = patch.get("asr_model")
        if next_model is not None:
            self._ensure_provider_registered(str(next_model))
        return self.config_store.update(patch, persist=persist)

    def switch_provider(self, provider: str, *, persist: bool = True) -> dict[str, Any]:
        """Switch the active ASR provider."""

        self._ensure_provider_registered(provider)
        return self.config_store.update({"asr_model": provider}, persist=persist)

    def list_providers(self) -> list[dict[str, Any]]:
        """Return registered provider metadata plus health/config state."""

        config = self.config_store.read()
        active_provider = self._active_provider(config)
        providers: list[dict[str, Any]] = []

        for name in ASRFactory.available():
            metadata = ASRFactory.metadata(name)
            provider_config = self._provider_config(config, name)
            health = self._provider_health(name, provider_config)
            providers.append(
                {
                    "name": metadata.name,
                    "display_name": metadata.display_name,
                    "provider_type": metadata.provider_type,
                    "description": metadata.description,
                    "active": name == active_provider,
                    "available": health["available"],
                    "reason": health["reason"],
                    "supports_backend_transcription": metadata.supports_backend_transcription,
                    "supports_browser_streaming": metadata.supports_browser_streaming,
                    "config": self._public_config(provider_config),
                }
            )

        return providers

    def health(self) -> dict[str, Any]:
        """Return active and all-provider ASR health state."""

        config = self.config_store.read()
        active_provider = self._active_provider(config)
        providers = self.list_providers()
        active = next(
            (provider for provider in providers if provider["name"] == active_provider),
            None,
        )
        return {
            "active_provider": active_provider,
            "active_available": bool(active and active["available"]),
            "providers": providers,
        }

    async def transcribe_audio(
        self,
        audio: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe uploaded audio with the selected backend-capable provider."""

        config = self.config_store.read()
        provider_name = provider or self._active_provider(config)
        self._ensure_provider_registered(provider_name)
        metadata = ASRFactory.metadata(provider_name)
        if not metadata.supports_backend_transcription:
            raise ASRProviderUnavailableError(
                f"ASR provider '{provider_name}' does not support backend transcription"
            )

        provider_config = self._provider_config(config, provider_name)
        asr = ASRFactory.create(provider_name, **provider_config)
        health = asr.health()
        if not health.available:
            raise ASRProviderUnavailableError(health.reason or f"{provider_name} is unavailable")

        text = await asr.async_transcribe_audio(
            audio,
            filename=filename,
            content_type=content_type,
        )
        return {
            "provider": provider_name,
            "text": text,
        }

    def _provider_health(self, name: str, provider_config: dict[str, Any]) -> dict[str, Any]:
        try:
            health = ASRFactory.create(name, **provider_config).health()
        except Exception as error:  # noqa: BLE001
            return {"available": False, "reason": str(error)}
        return {"available": health.available, "reason": health.reason}

    def _active_provider(self, config: dict[str, Any]) -> str:
        provider = str(config.get("asr_model") or "web_speech_api")
        self._ensure_provider_registered(provider)
        return provider

    def _provider_config(self, config: dict[str, Any], provider: str) -> dict[str, Any]:
        value = config.get(provider)
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _ensure_provider_registered(self, provider: str) -> None:
        if provider not in ASRFactory.available():
            raise ASRConfigError(
                f"Unknown ASR provider: {provider!r}. Available: {ASRFactory.available()}"
            )

    def _public_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return config safe for API responses."""

        safe: dict[str, Any] = {}
        for key, value in config.items():
            if key.lower() in SENSITIVE_CONFIG_KEYS:
                safe[key] = SENSITIVE_CONFIG_MASK if value else value
            elif isinstance(value, dict):
                safe[key] = self._public_config(value)
            else:
                safe[key] = value
        return safe

    def _strip_masked_sensitive_values(self, config: dict[str, Any]) -> dict[str, Any]:
        """Remove masked secrets from incoming API patches."""

        cleaned: dict[str, Any] = {}
        for key, value in config.items():
            if key.lower() in SENSITIVE_CONFIG_KEYS and value == SENSITIVE_CONFIG_MASK:
                continue
            if isinstance(value, dict):
                cleaned[key] = self._strip_masked_sensitive_values(value)
            else:
                cleaned[key] = value
        return cleaned
