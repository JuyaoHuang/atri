"""OpenAI Whisper-compatible cloud ASR provider."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

from src.asr.exceptions import ASRProviderUnavailableError
from src.asr.factory import ASRFactory, ASRProviderMetadata
from src.asr.interface import ASRHealth, ASRInterface


@ASRFactory.register(
    "openai_whisper",
    metadata=ASRProviderMetadata(
        name="openai_whisper",
        display_name="OpenAI Whisper",
        provider_type="cloud",
        supports_backend_transcription=True,
        supports_browser_streaming=False,
        description="Cloud audio transcription through the OpenAI-compatible SDK.",
    ),
)
class OpenAIWhisperASR(ASRInterface):
    """Cloud transcription provider with lazy OpenAI client creation."""

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.model = str(config.get("model") or "whisper-1")
        self.api_key = str(config.get("api_key") or "")
        self.base_url = str(config.get("base_url") or "")
        self.language = str(config.get("language") or "")
        self.prompt = str(config.get("prompt") or "")

    def health(self) -> ASRHealth:
        if importlib.util.find_spec("openai") is None:
            return ASRHealth(False, "Python package 'openai' is not installed")
        if not self.api_key or self.api_key.startswith("${"):
            return ASRHealth(False, "openai_whisper.api_key is not configured")
        if not self.model:
            return ASRHealth(False, "openai_whisper.model is not configured")
        return ASRHealth(True)

    def transcribe_np(self, audio: Any) -> str:
        raise ASRProviderUnavailableError(
            "openai_whisper accepts uploaded audio files, not numpy arrays"
        )

    async def async_transcribe_audio(
        self,
        audio: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        health = self.health()
        if not health.available:
            raise ASRProviderUnavailableError(health.reason or "openai_whisper is unavailable")

        suffix = Path(filename or "recording.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
            temp_audio.write(audio)
            temp_audio.flush()
            return await self._transcribe_file(temp_audio.name)

    async def _transcribe_file(self, path: str) -> str:
        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = AsyncOpenAI(**client_kwargs)

        kwargs: dict[str, Any] = {"model": self.model}
        if self.language:
            kwargs["language"] = self.language
        if self.prompt:
            kwargs["prompt"] = self.prompt

        with Path(path).open("rb") as audio_file:
            response = await client.audio.transcriptions.create(file=audio_file, **kwargs)

        return str(getattr(response, "text", "") or "")
