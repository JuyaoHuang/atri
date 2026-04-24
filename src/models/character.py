"""Character API schemas used by Phase 7."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CharacterSummary(BaseModel):
    """Character summary returned by list endpoints."""

    character_id: str
    name: str
    avatar: str | None = None
    avatar_url: str | None = None
    greeting: str | None = None
    description: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    is_system: bool = True


class CharacterDetail(CharacterSummary):
    """Character detail with full system prompt."""

    system_prompt: str


class CharacterCreateRequest(BaseModel):
    """Payload for creating a character."""

    character_id: str | None = Field(default=None, max_length=64)
    name: str = Field(..., min_length=1, max_length=50)
    greeting: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=200)
    system_prompt: str = Field(..., min_length=1, max_length=4000)


class CharacterUpdateRequest(BaseModel):
    """Payload for updating a character."""

    name: str | None = Field(default=None, min_length=1, max_length=50)
    greeting: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=200)
    system_prompt: str | None = Field(default=None, min_length=1, max_length=4000)


class AvatarUploadResponse(BaseModel):
    """Response payload for avatar upload."""

    character_id: str
    avatar: str
    avatar_url: str
