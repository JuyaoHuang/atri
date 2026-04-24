"""Character storage backed by persona markdown files and managed avatars."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml
from fastapi import UploadFile

from src.agent.persona import Persona, load_persona_from_path

MANAGED_BY_ATRI = "atri"
MAX_AVATAR_SIZE_BYTES = 2 * 1024 * 1024
ALLOWED_AVATAR_CONTENT_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PERSONA_DIR = _PROJECT_ROOT / "prompts" / "persona"
_DEFAULT_AVATAR_DIR = _PROJECT_ROOT / "data" / "avatars"


class CharacterStorageError(Exception):
    """Base exception for character storage operations."""


class CharacterNotFoundError(CharacterStorageError):
    """Raised when a character file does not exist."""


class CharacterNameConflictError(CharacterStorageError):
    """Raised when a character name duplicates another character."""


class CharacterSystemDeleteError(CharacterStorageError):
    """Raised when attempting to delete a protected system character."""


class AvatarValidationError(CharacterStorageError):
    """Raised when avatar upload validation fails."""


@dataclass(frozen=True)
class CharacterRecord:
    """Character record loaded from persona markdown."""

    character_id: str
    name: str
    avatar: str | None
    greeting: str | None
    description: str | None
    system_prompt: str
    created_at: str | None
    updated_at: str | None
    managed_by: str | None

    @property
    def is_system(self) -> bool:
        return self.managed_by != MANAGED_BY_ATRI


def get_default_character_persona_dir() -> Path:
    """Return the default persona directory used by the backend."""

    return _DEFAULT_PERSONA_DIR


def get_default_character_avatar_dir() -> Path:
    """Return the default managed avatar directory used by the backend."""

    return _DEFAULT_AVATAR_DIR


def _now_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _slugify_character_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized


class CharacterStorage:
    """Read and write characters using persona markdown files."""

    def __init__(self, persona_dir: Path | None = None, avatar_dir: Path | None = None) -> None:
        self.persona_dir = persona_dir or get_default_character_persona_dir()
        self.avatar_dir = avatar_dir or get_default_character_avatar_dir()

        self.persona_dir.mkdir(parents=True, exist_ok=True)
        self.avatar_dir.mkdir(parents=True, exist_ok=True)

    def list_characters(self) -> list[CharacterRecord]:
        """List all character records sorted by name."""

        records = [self.get_character(path.stem) for path in sorted(self.persona_dir.glob("*.md"))]
        return sorted(records, key=lambda record: (record.name.casefold(), record.character_id))

    def get_character(self, character_id: str) -> CharacterRecord:
        """Load a single character by its identifier."""

        path = self._character_path(character_id)
        if not path.is_file():
            raise CharacterNotFoundError(f"Character '{character_id}' not found")

        persona = load_persona_from_path(path)
        return self._record_from_persona(persona)

    def create_character(
        self,
        *,
        character_id: str | None,
        name: str,
        greeting: str | None,
        description: str | None,
        system_prompt: str,
    ) -> CharacterRecord:
        """Create a new managed character."""

        clean_name = name.strip()
        clean_system_prompt = system_prompt.strip()
        clean_greeting = greeting.strip() if greeting else None
        clean_description = description.strip() if description else None

        if not clean_name:
            raise CharacterStorageError("Character name cannot be empty")
        if not clean_system_prompt:
            raise CharacterStorageError("System prompt cannot be empty")

        self._ensure_unique_name(clean_name)
        resolved_id = self._resolve_character_id(character_id, clean_name)
        now = _now_iso_z()

        record = CharacterRecord(
            character_id=resolved_id,
            name=clean_name,
            avatar=None,
            greeting=clean_greeting,
            description=clean_description,
            system_prompt=clean_system_prompt,
            created_at=now,
            updated_at=now,
            managed_by=MANAGED_BY_ATRI,
        )
        self._write_character(record)
        return record

    def update_character(
        self,
        character_id: str,
        *,
        name: str | None = None,
        greeting: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
    ) -> CharacterRecord:
        """Update an existing character."""

        current = self.get_character(character_id)

        next_name = name.strip() if name is not None else current.name
        next_greeting = greeting.strip() if greeting is not None else current.greeting
        next_description = description.strip() if description is not None else current.description
        next_system_prompt = (
            system_prompt.strip() if system_prompt is not None else current.system_prompt
        )

        if not next_name:
            raise CharacterStorageError("Character name cannot be empty")
        if not next_system_prompt:
            raise CharacterStorageError("System prompt cannot be empty")

        self._ensure_unique_name(next_name, ignore_character_id=character_id)

        record = CharacterRecord(
            character_id=current.character_id,
            name=next_name,
            avatar=current.avatar,
            greeting=next_greeting,
            description=next_description,
            system_prompt=next_system_prompt,
            created_at=current.created_at,
            updated_at=_now_iso_z(),
            managed_by=current.managed_by,
        )
        self._write_character(record)
        return record

    def delete_character(self, character_id: str) -> None:
        """Delete a managed character and its uploaded avatar, if any."""

        record = self.get_character(character_id)
        if record.is_system:
            raise CharacterSystemDeleteError(
                f"Character '{character_id}' is a protected system character and cannot be deleted"
            )

        avatar_path = self._managed_avatar_path(record.avatar)
        self._character_path(character_id).unlink(missing_ok=True)
        if avatar_path is not None and avatar_path.is_file():
            avatar_path.unlink(missing_ok=True)

    async def save_avatar(self, character_id: str, file: UploadFile) -> CharacterRecord:
        """Validate and store an uploaded avatar for a character."""

        current = self.get_character(character_id)
        extension = self._resolve_avatar_extension(file)
        payload = await file.read()

        if not payload:
            raise AvatarValidationError("Avatar file cannot be empty")
        if len(payload) > MAX_AVATAR_SIZE_BYTES:
            raise AvatarValidationError("Avatar file size must be smaller than 2MB")

        filename = f"{character_id}-{uuid4().hex[:8]}{extension}"
        target_path = self.avatar_dir / filename
        target_path.write_bytes(payload)

        old_avatar_path = self._managed_avatar_path(current.avatar)
        if old_avatar_path is not None and old_avatar_path.is_file():
            old_avatar_path.unlink(missing_ok=True)

        record = CharacterRecord(
            character_id=current.character_id,
            name=current.name,
            avatar=filename,
            greeting=current.greeting,
            description=current.description,
            system_prompt=current.system_prompt,
            created_at=current.created_at,
            updated_at=_now_iso_z(),
            managed_by=current.managed_by,
        )
        self._write_character(record)
        return record

    def build_avatar_url(self, avatar: str | None, base_url: str) -> str | None:
        """Build an absolute backend avatar URL for managed avatars."""

        managed_path = self._managed_avatar_path(avatar)
        if avatar is None or managed_path is None or not managed_path.is_file():
            return None

        return f"{base_url.rstrip('/')}/api/assets/avatars/{avatar}"

    def _character_path(self, character_id: str) -> Path:
        return self.persona_dir / f"{character_id}.md"

    def _record_from_persona(self, persona: Persona) -> CharacterRecord:
        return CharacterRecord(
            character_id=persona.character_id,
            name=persona.name,
            avatar=persona.avatar,
            greeting=persona.greeting,
            description=persona.description,
            system_prompt=persona.system_prompt,
            created_at=persona.created_at,
            updated_at=persona.updated_at,
            managed_by=persona.managed_by,
        )

    def _ensure_unique_name(self, name: str, ignore_character_id: str | None = None) -> None:
        for record in self.list_characters():
            if ignore_character_id is not None and record.character_id == ignore_character_id:
                continue
            if record.name.casefold() == name.casefold():
                raise CharacterNameConflictError(f"Character name '{name}' already exists")

    def _resolve_character_id(self, requested_id: str | None, name: str) -> str:
        raw_id = requested_id.strip() if requested_id else name
        base_id = _slugify_character_id(raw_id)
        if not base_id:
            base_id = f"character-{uuid4().hex[:8]}"

        candidate = base_id
        suffix = 2
        while self._character_path(candidate).exists():
            candidate = f"{base_id}-{suffix}"
            suffix += 1
        return candidate

    def _write_character(self, record: CharacterRecord) -> None:
        metadata = {
            "name": record.name,
            "avatar": record.avatar,
            "greeting": record.greeting,
            "description": record.description,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "managed_by": record.managed_by,
        }
        clean_metadata = {key: value for key, value in metadata.items() if value is not None}
        frontmatter = yaml.safe_dump(clean_metadata, allow_unicode=True, sort_keys=False).strip()
        body = record.system_prompt.strip()
        content = f"---\n{frontmatter}\n---\n\n{body}\n"
        self._character_path(record.character_id).write_text(content, encoding="utf-8")

    def _resolve_avatar_extension(self, file: UploadFile) -> str:
        content_type = file.content_type or ""
        if content_type in ALLOWED_AVATAR_CONTENT_TYPES:
            return ALLOWED_AVATAR_CONTENT_TYPES[content_type]

        filename = file.filename or ""
        extension = Path(filename).suffix.lower()
        if extension in {".png", ".jpg", ".jpeg", ".webp"}:
            return ".jpg" if extension == ".jpeg" else extension

        raise AvatarValidationError("Only PNG/JPG/WEBP avatars are supported")

    def _managed_avatar_path(self, avatar: str | None) -> Path | None:
        if not avatar:
            return None

        path = self.avatar_dir / avatar
        return path if path.parent == self.avatar_dir else None
