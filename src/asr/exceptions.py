"""ASR-specific exception hierarchy."""

from __future__ import annotations


class ASRError(Exception):
    """Base exception for ASR operations."""


class ASRConfigError(ASRError):
    """Raised when ASR configuration is invalid."""


class ASRProviderUnavailableError(ASRError):
    """Raised when a provider cannot run because dependencies or config are missing."""


class ASRTranscriptionError(ASRError):
    """Raised when transcription fails after a provider was selected."""
