"""ASR provider registrations."""

from . import faster_whisper, openai_whisper, web_speech_api, whisper_cpp

__all__ = [
    "faster_whisper",
    "openai_whisper",
    "web_speech_api",
    "whisper_cpp",
]
