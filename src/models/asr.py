"""ASR API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ASRProviderStatus(BaseModel):
    """Provider metadata plus runtime availability."""

    name: str
    display_name: str
    provider_type: str
    description: str
    active: bool
    available: bool
    reason: str | None = None
    supports_backend_transcription: bool
    supports_browser_streaming: bool
    config: dict[str, Any] = Field(default_factory=dict)


class ASRConfigResponse(BaseModel):
    """OLV-shaped ASR config response."""

    config: dict[str, Any]
    providers: list[ASRProviderStatus] = Field(default_factory=list)


class ASRHealthResponse(BaseModel):
    """ASR health response."""

    active_provider: str
    active_available: bool
    providers: list[ASRProviderStatus] = Field(default_factory=list)


class ASRProviderSwitchRequest(BaseModel):
    """Switch active provider payload."""

    provider: str = Field(..., min_length=1)


class ASRTranscriptionResponse(BaseModel):
    """Transcription result response."""

    provider: str
    text: str
