"""Pydantic request and response schemas."""

from .character import (
    AvatarUploadResponse,
    CharacterCreateRequest,
    CharacterDetail,
    CharacterSummary,
    CharacterUpdateRequest,
)

__all__ = [
    "AvatarUploadResponse",
    "CharacterCreateRequest",
    "CharacterDetail",
    "CharacterSummary",
    "CharacterUpdateRequest",
]
