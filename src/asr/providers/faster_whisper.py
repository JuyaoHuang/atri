"""faster-whisper ASR provider.

The numpy-array transcription path follows Open-LLM-VTuber's
`faster_whisper_asr.py`: lazy-create `WhisperModel`, call `transcribe`
with beam search, optional language and prompt, and concatenate segment
text.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

from src.asr.exceptions import ASRProviderUnavailableError
from src.asr.factory import ASRFactory, ASRProviderMetadata
from src.asr.interface import ASRHealth, ASRInterface


@ASRFactory.register(
    "faster_whisper",
    metadata=ASRProviderMetadata(
        name="faster_whisper",
        display_name="Faster Whisper",
        provider_type="local",
        supports_backend_transcription=True,
        supports_browser_streaming=False,
        description="Local faster-whisper transcription compatible with OLV flow.",
    ),
)
class FasterWhisperASR(ASRInterface):
    """Local faster-whisper provider with lazy optional dependency loading."""

    BEAM_SEARCH = True

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.model_path = str(config.get("model_path") or "large-v3-turbo")
        self.download_root = config.get("download_root") or None
        self.language = config.get("language") or None
        self.device = str(config.get("device") or "auto")
        self.compute_type = str(config.get("compute_type") or "int8")
        self.prompt = config.get("prompt") or None
        self._model: Any | None = None

    def health(self) -> ASRHealth:
        if importlib.util.find_spec("faster_whisper") is None:
            return ASRHealth(False, "Python package 'faster_whisper' is not installed")
        if not self.model_path:
            return ASRHealth(False, "faster_whisper.model_path is not configured")
        return ASRHealth(True)

    def transcribe_np(self, audio: Any) -> str:
        model = self._get_model()
        kwargs: dict[str, Any] = {
            "beam_size": 5 if self.BEAM_SEARCH else 1,
            "language": self.language if self.language and self.language != "auto" else None,
            "condition_on_previous_text": False,
        }
        if self.prompt:
            kwargs["initial_prompt"] = self.prompt

        segments, _info = model.transcribe(audio, **kwargs)
        return "".join(segment.text for segment in segments)

    async def async_transcribe_audio(
        self,
        audio: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        """Transcribe uploaded audio.

        WAV uploads use the OLV numpy-array path. Other browser formats are
        passed to faster-whisper through a temporary file, which keeps the
        provider usable with MediaRecorder while preserving `transcribe_np`
        as the core local-provider contract.
        """

        if self._looks_like_wav(audio, filename=filename, content_type=content_type):
            return await super().async_transcribe_audio(
                audio, filename=filename, content_type=content_type
            )

        suffix = Path(filename or "recording.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
            temp_audio.write(audio)
            temp_audio.flush()
            return await self._async_transcribe_file_path(temp_audio.name)

    async def _async_transcribe_file_path(self, path: str) -> str:
        import asyncio

        return await asyncio.to_thread(self._transcribe_file_path, path)

    def _transcribe_file_path(self, path: str) -> str:
        model = self._get_model()
        kwargs: dict[str, Any] = {
            "beam_size": 5 if self.BEAM_SEARCH else 1,
            "language": self.language if self.language and self.language != "auto" else None,
            "condition_on_previous_text": False,
        }
        if self.prompt:
            kwargs["initial_prompt"] = self.prompt

        segments, _info = model.transcribe(path, **kwargs)
        return "".join(segment.text for segment in segments)

    def _get_model(self) -> Any:
        health = self.health()
        if not health.available:
            raise ASRProviderUnavailableError(health.reason or "faster_whisper is unavailable")

        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(
                model_size_or_path=self.model_path,
                download_root=self.download_root,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model
