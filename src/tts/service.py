"""TTS application service."""

from __future__ import annotations

from typing import Any

from . import providers as _providers  # noqa: F401
from .config import TTSConfigStore
from .exceptions import TTSConfigError, TTSProviderUnavailableError, TTSSynthesisError
from .factory import TTSFactory
from .interface import TTSVoice

SENSITIVE_CONFIG_KEYS = {"api_key", "token", "secret", "password"}
SENSITIVE_CONFIG_MASK = "********"
PROVIDER_WRITE_ALLOWLISTS: dict[str, set[str]] = {
    "edge_tts": {"voice", "rate"},
    "gpt_sovits_tts": set(),
    "siliconflow_tts": {"default_voice", "stream"},
    "cosyvoice3_tts": {"sft_dropdown", "stream", "speed"},
}


class TTSService:
    """Coordinate TTS config, provider health, switching, voices, and synthesis."""

    def __init__(self, config_store: TTSConfigStore) -> None:
        self.config_store = config_store

    def get_config(self) -> dict[str, Any]:
        """Return persisted OLV-shaped TTS config."""

        return self._public_config(self.config_store.read())

    def update_config(self, patch: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
        """Merge and persist a partial OLV-shaped TTS config update."""

        patch = self._strip_masked_sensitive_values(patch)
        patch = self._strip_forbidden_provider_writes(patch)
        next_model = patch.get("tts_model")
        if next_model is not None:
            self._ensure_provider_registered(str(next_model))
        return self.config_store.update(patch, persist=persist)

    def switch_provider(self, provider: str, *, persist: bool = True) -> dict[str, Any]:
        """Switch the active TTS provider."""

        self._ensure_provider_registered(provider)
        return self.config_store.update({"tts_model": provider}, persist=persist)

    def list_providers(self) -> list[dict[str, Any]]:
        """Return registered provider metadata plus health/config state."""

        config = self.config_store.read()
        active_provider = self._active_provider(config)
        providers: list[dict[str, Any]] = []

        for name in TTSFactory.available():
            metadata = TTSFactory.metadata(name)
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
                    "supports_streaming": metadata.supports_streaming,
                    "media_type": self._provider_media_type(name, provider_config),
                    "config": self._public_config(provider_config),
                }
            )

        return providers

    def health(self) -> dict[str, Any]:
        """Return active and all-provider TTS health state."""

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

    async def get_voices(self, provider: str | None = None) -> dict[str, Any]:
        """Return voices for the active or requested TTS provider."""

        config = self.config_store.read()
        provider_name = provider or self._active_provider(config)
        tts = self._create_provider(config, provider_name)
        voices = await tts.get_voices()
        return {
            "provider": provider_name,
            "voices": [self._voice_to_dict(voice) for voice in voices],
        }

    async def synthesize(
        self,
        text: str,
        *,
        provider: str | None = None,
        voice_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesize text with the selected provider and return audio bytes."""

        text = text.strip()
        if not text:
            raise TTSSynthesisError("TTS text must not be empty")

        config = self.config_store.read()
        provider_name = provider or self._active_provider(config)
        tts = self._create_provider(config, provider_name)
        health = tts.health()
        if not health.available:
            raise TTSProviderUnavailableError(health.reason or f"{provider_name} is unavailable")

        audio = await tts.synthesize(text, voice_id=voice_id, **(options or {}))
        return {
            "provider": provider_name,
            "audio": audio,
            "media_type": tts.media_type,
        }

    def _create_provider(self, config: dict[str, Any], provider: str) -> Any:
        self._ensure_provider_registered(provider)
        provider_config = self._provider_config(config, provider)
        return TTSFactory.create(provider, **provider_config)

    def _provider_health(self, name: str, provider_config: dict[str, Any]) -> dict[str, Any]:
        try:
            health = TTSFactory.create(name, **provider_config).health()
        except Exception as error:  # noqa: BLE001
            return {"available": False, "reason": str(error)}
        return {"available": health.available, "reason": health.reason}

    def _provider_media_type(self, name: str, provider_config: dict[str, Any]) -> str:
        try:
            return TTSFactory.create(name, **provider_config).media_type
        except Exception:
            return TTSFactory.metadata(name).media_type

    def _active_provider(self, config: dict[str, Any]) -> str:
        provider = str(config.get("tts_model") or "edge_tts")
        self._ensure_provider_registered(provider)
        return provider

    def _provider_config(self, config: dict[str, Any], provider: str) -> dict[str, Any]:
        value = config.get(provider)
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _ensure_provider_registered(self, provider: str) -> None:
        if provider not in TTSFactory.available():
            raise TTSConfigError(
                f"Unknown TTS provider: {provider!r}. Available: {TTSFactory.available()}"
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

    def _strip_forbidden_provider_writes(self, config: dict[str, Any]) -> dict[str, Any]:
        """Remove provider fields that must remain backend-controlled."""

        cleaned: dict[str, Any] = {}
        for key, value in config.items():
            if not isinstance(value, dict):
                cleaned[key] = value
                continue

            allowed = PROVIDER_WRITE_ALLOWLISTS.get(key)
            if allowed is None:
                provider_config = dict(value)
            else:
                provider_config = {
                    field: field_value
                    for field, field_value in value.items()
                    if field in allowed
                }

            if provider_config:
                cleaned[key] = provider_config
        return cleaned

    def _voice_to_dict(self, voice: TTSVoice) -> dict[str, Any]:
        return {
            "id": voice.id,
            "name": voice.name,
            "language": voice.language,
            "gender": voice.gender,
            "description": voice.description,
            "preview_url": voice.preview_url,
        }
