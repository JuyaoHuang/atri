"""Browser Web Speech API provider marker.

The implementation follows AIRI's split: Web Speech API runs in the browser
with live recognition, while the backend only exposes configuration and
provider status.
"""

from __future__ import annotations

from typing import Any

from src.asr.exceptions import ASRProviderUnavailableError
from src.asr.factory import ASRFactory, ASRProviderMetadata
from src.asr.interface import ASRHealth, ASRInterface


@ASRFactory.register(
    "web_speech_api",
    metadata=ASRProviderMetadata(
        name="web_speech_api",
        display_name="Web Speech API",
        provider_type="browser",
        supports_backend_transcription=False,
        supports_browser_streaming=True,
        description="Browser-native Web Speech API provider; transcription runs in frontend.",
    ),
)
class WebSpeechAPIASR(ASRInterface):
    """Configuration-only provider for browser Web Speech API."""

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.language = str(config.get("language") or "zh-CN")
        self.continuous = bool(config.get("continuous", True))
        self.interim_results = bool(config.get("interim_results", True))
        self.max_alternatives = int(config.get("max_alternatives", 1))

    def health(self) -> ASRHealth:
        return ASRHealth(True, "Browser availability is checked in the frontend")

    def transcribe_np(self, audio: Any) -> str:
        raise ASRProviderUnavailableError(
            "Web Speech API runs in the browser and does not support backend audio uploads"
        )

    async def async_transcribe_audio(
        self,
        audio: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        raise ASRProviderUnavailableError(
            "Web Speech API runs in the browser and does not support backend audio uploads"
        )
