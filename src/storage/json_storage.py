"""JSON file-based chat storage implementation."""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.memory._io_utils import atomic_replace
from src.storage.interface import ChatStorageInterface


class JSONChatStorage(ChatStorageInterface):
    """JSON file-based chat storage with file system persistence."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def _get_user_dir(self, user_id: str, character_id: str) -> Path:
        """Get user-character directory path."""
        return self.base_path / user_id / character_id

    def _get_index_path(self, user_id: str, character_id: str) -> Path:
        """Get index.json path."""
        return self._get_user_dir(user_id, character_id) / "index.json"

    def _get_session_path(self, user_id: str, character_id: str, chat_id: str) -> Path:
        """Get session file path."""
        return self._get_user_dir(user_id, character_id) / "sessions" / f"{chat_id}.json"

    async def _read_json(self, path: Path) -> dict | list | None:
        """Read JSON file asynchronously."""

        def _read():
            if not path.exists():
                return None
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

    async def _write_json(self, path: Path, data: dict | list) -> None:
        """Write JSON file atomically."""

        def _write():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            atomic_replace(tmp, path)

        await asyncio.to_thread(_write)

    async def _load_index(self, user_id: str, character_id: str) -> dict:
        """Load index.json or return empty structure."""
        index_path = self._get_index_path(user_id, character_id)
        data = await self._read_json(index_path)
        if isinstance(data, dict):
            return data
        return {"chats": []}

    async def _save_index(self, user_id: str, character_id: str, index: dict) -> None:
        """Save index.json atomically."""
        index_path = self._get_index_path(user_id, character_id)
        await self._write_json(index_path, index)

    def _generate_chat_id(self) -> str:
        """Generate chat ID in format: YYYYMMDD_uuid8."""
        date_str = datetime.now(UTC).strftime("%Y%m%d")
        uuid_str = str(uuid.uuid4())[:8]
        return f"{date_str}_{uuid_str}"

    async def create_chat(self, user_id: str, character_id: str, title: str) -> dict:
        """Create a new chat session."""
        chat_id = self._generate_chat_id()
        now = datetime.now(UTC).isoformat()

        chat_meta = {
            "id": chat_id,
            "title": title,
            "character_id": character_id,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

        # Update index
        index = await self._load_index(user_id, character_id)
        index["chats"].append(chat_meta)
        await self._save_index(user_id, character_id, index)

        # Create empty session file
        session_path = self._get_session_path(user_id, character_id, chat_id)
        await self._write_json(session_path, {"messages": []})

        return chat_meta

    async def list_chats(self, user_id: str, character_id: str | None = None) -> list[dict]:
        """List user's chat sessions, sorted by updated_at descending."""
        if character_id:
            index = await self._load_index(user_id, character_id)
            chats = index["chats"]
        else:
            # Aggregate across all characters
            chats = []
            user_dir = self.base_path / user_id
            if user_dir.exists():
                for char_dir in user_dir.iterdir():
                    if char_dir.is_dir():
                        index = await self._load_index(user_id, char_dir.name)
                        chats.extend(index["chats"])

        # Sort by updated_at descending
        chats.sort(key=lambda c: c["updated_at"], reverse=True)
        return chats

    async def get_chat(self, chat_id: str) -> dict | None:
        """Get chat metadata by ID (requires scanning all user directories)."""
        # Note: This is inefficient for production use. Phase 7 database will fix this.
        if not self.base_path.is_dir():
            return None
        for user_dir in self.base_path.iterdir():
            if not user_dir.is_dir():
                continue
            for char_dir in user_dir.iterdir():
                if not char_dir.is_dir():
                    continue
                index = await self._load_index(user_dir.name, char_dir.name)
                for chat in index["chats"]:
                    if chat["id"] == chat_id:
                        return chat
        return None

    async def update_chat(self, chat_id: str, **kwargs: str) -> dict:
        """Update chat metadata (title, etc.)."""
        # Find the chat in index
        if not self.base_path.is_dir():
            raise ValueError(f"Chat {chat_id} not found")
        for user_dir in self.base_path.iterdir():
            if not user_dir.is_dir():
                continue
            for char_dir in user_dir.iterdir():
                if not char_dir.is_dir():
                    continue
                user_id = user_dir.name
                character_id = char_dir.name
                index = await self._load_index(user_id, character_id)

                for chat in index["chats"]:
                    if chat["id"] == chat_id:
                        # Update fields
                        chat.update(kwargs)
                        chat["updated_at"] = datetime.now(UTC).isoformat()
                        await self._save_index(user_id, character_id, index)
                        return chat

        raise ValueError(f"Chat {chat_id} not found")

    async def delete_chat(self, chat_id: str) -> None:
        """Delete chat session."""
        # Find and remove from index
        if not self.base_path.is_dir():
            raise ValueError(f"Chat {chat_id} not found")
        for user_dir in self.base_path.iterdir():
            if not user_dir.is_dir():
                continue
            for char_dir in user_dir.iterdir():
                if not char_dir.is_dir():
                    continue
                user_id = user_dir.name
                character_id = char_dir.name
                index = await self._load_index(user_id, character_id)

                for i, chat in enumerate(index["chats"]):
                    if chat["id"] == chat_id:
                        # Remove from index
                        index["chats"].pop(i)
                        await self._save_index(user_id, character_id, index)

                        # Delete session file
                        session_path = self._get_session_path(user_id, character_id, chat_id)

                        def _delete(path=session_path):
                            if path.exists():
                                path.unlink()

                        await asyncio.to_thread(_delete)
                        return

        raise ValueError(f"Chat {chat_id} not found")

    async def append_message(
        self, chat_id: str, role: str, content: str, name: str | None = None
    ) -> dict:
        """Append message to chat session."""
        # Find chat metadata
        chat_meta = await self.get_chat(chat_id)
        if not chat_meta:
            raise ValueError(f"Chat {chat_id} not found")

        user_id = None
        character_id = chat_meta["character_id"]

        # Find user_id by scanning
        for user_dir in self.base_path.iterdir():
            if not user_dir.is_dir():
                continue
            char_dir = user_dir / character_id
            if char_dir.exists():
                index = await self._load_index(user_dir.name, character_id)
                if any(c["id"] == chat_id for c in index["chats"]):
                    user_id = user_dir.name
                    break

        if not user_id:
            raise ValueError(f"Chat {chat_id} not found")

        # Load session
        session_path = self._get_session_path(user_id, character_id, chat_id)
        session_data = await self._read_json(session_path)
        if not isinstance(session_data, dict):
            session_data = {"messages": []}

        # Append message
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if name:
            message["name"] = name

        session_data["messages"].append(message)
        await self._write_json(session_path, session_data)

        # Update index metadata
        index = await self._load_index(user_id, character_id)
        for chat in index["chats"]:
            if chat["id"] == chat_id:
                chat["message_count"] = len(session_data["messages"])
                chat["updated_at"] = message["timestamp"]
                break
        await self._save_index(user_id, character_id, index)

        return message

    async def get_messages(self, chat_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get chat message history with pagination."""
        chat_meta = await self.get_chat(chat_id)
        if not chat_meta:
            raise ValueError(f"Chat {chat_id} not found")

        user_id = None
        character_id = chat_meta["character_id"]

        # Find user_id
        for user_dir in self.base_path.iterdir():
            if not user_dir.is_dir():
                continue
            char_dir = user_dir / character_id
            if char_dir.exists():
                index = await self._load_index(user_dir.name, character_id)
                if any(c["id"] == chat_id for c in index["chats"]):
                    user_id = user_dir.name
                    break

        if not user_id:
            raise ValueError(f"Chat {chat_id} not found")

        # Load session
        session_path = self._get_session_path(user_id, character_id, chat_id)
        session_data = await self._read_json(session_path)
        if not isinstance(session_data, dict):
            return []

        messages = session_data.get("messages", [])
        return messages[offset : offset + limit]
