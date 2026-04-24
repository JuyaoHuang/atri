"""Character management REST API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from ..models.character import (
    AvatarUploadResponse,
    CharacterCreateRequest,
    CharacterDetail,
    CharacterSummary,
    CharacterUpdateRequest,
)
from ..storage.character_storage import (
    AvatarValidationError,
    CharacterNameConflictError,
    CharacterNotFoundError,
    CharacterStorage,
    CharacterStorageError,
    CharacterSystemDeleteError,
)

router = APIRouter(prefix="/api/characters", tags=["characters"])


def get_character_storage(request: Request) -> CharacterStorage:
    """Return app-scoped character storage, creating a default one if needed."""

    storage = getattr(request.app.state, "character_storage", None)
    if storage is None:
        storage = CharacterStorage()
        request.app.state.character_storage = storage
    return storage


CharacterStorageDep = Annotated[CharacterStorage, Depends(get_character_storage)]
AvatarFile = Annotated[UploadFile, File(...)]


def _serialize_character(
    record,
    request: Request,
    storage: CharacterStorage,
) -> CharacterDetail:
    avatar_url = storage.build_avatar_url(record.avatar, str(request.base_url))
    return CharacterDetail(
        character_id=record.character_id,
        name=record.name,
        avatar=record.avatar,
        avatar_url=avatar_url,
        greeting=record.greeting,
        description=record.description,
        created_at=record.created_at,
        updated_at=record.updated_at,
        is_system=record.is_system,
        system_prompt=record.system_prompt,
    )


def _handle_character_error(error: Exception) -> HTTPException:
    if isinstance(error, CharacterNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, CharacterNameConflictError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, CharacterSystemDeleteError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, AvatarValidationError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, CharacterStorageError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail="Character storage operation failed")


@router.get("", response_model=list[CharacterSummary])
async def list_characters(
    request: Request,
    storage: CharacterStorageDep,
) -> list[CharacterSummary]:
    """List all characters without changing the existing read API shape."""

    records = storage.list_characters()
    return [
        CharacterSummary(
            character_id=record.character_id,
            name=record.name,
            avatar=record.avatar,
            avatar_url=storage.build_avatar_url(record.avatar, str(request.base_url)),
            greeting=record.greeting,
            description=record.description,
            created_at=record.created_at,
            updated_at=record.updated_at,
            is_system=record.is_system,
        )
        for record in records
    ]


@router.get("/{character_id}", response_model=CharacterDetail)
async def get_character(
    character_id: str,
    request: Request,
    storage: CharacterStorageDep,
) -> CharacterDetail:
    """Get one character with its full system prompt."""

    try:
        record = storage.get_character(character_id)
    except Exception as error:
        raise _handle_character_error(error) from error

    return _serialize_character(record, request, storage)


@router.post("", response_model=CharacterDetail, status_code=status.HTTP_201_CREATED)
async def create_character(
    payload: CharacterCreateRequest,
    request: Request,
    storage: CharacterStorageDep,
) -> CharacterDetail:
    """Create a new character."""

    try:
        record = storage.create_character(
            character_id=payload.character_id,
            name=payload.name,
            greeting=payload.greeting,
            description=payload.description,
            system_prompt=payload.system_prompt,
        )
    except Exception as error:
        raise _handle_character_error(error) from error

    return _serialize_character(record, request, storage)


@router.put("/{character_id}", response_model=CharacterDetail)
async def update_character(
    character_id: str,
    payload: CharacterUpdateRequest,
    request: Request,
    storage: CharacterStorageDep,
) -> CharacterDetail:
    """Update an existing character."""

    try:
        record = storage.update_character(
            character_id,
            name=payload.name,
            greeting=payload.greeting,
            description=payload.description,
            system_prompt=payload.system_prompt,
        )
    except Exception as error:
        raise _handle_character_error(error) from error

    return _serialize_character(record, request, storage)


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(
    character_id: str,
    storage: CharacterStorageDep,
) -> Response:
    """Delete a managed character."""

    try:
        storage.delete_character(character_id)
    except Exception as error:
        raise _handle_character_error(error) from error

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{character_id}/avatar", response_model=AvatarUploadResponse)
async def upload_character_avatar(
    character_id: str,
    request: Request,
    avatar: AvatarFile,
    storage: CharacterStorageDep,
) -> AvatarUploadResponse:
    """Upload or replace a character avatar."""

    try:
        record = await storage.save_avatar(character_id, avatar)
    except Exception as error:
        raise _handle_character_error(error) from error

    avatar_url = storage.build_avatar_url(record.avatar, str(request.base_url))
    if avatar_url is None or record.avatar is None:
        raise HTTPException(status_code=500, detail="Character avatar URL could not be generated")

    return AvatarUploadResponse(
        character_id=record.character_id,
        avatar=record.avatar,
        avatar_url=avatar_url,
    )
