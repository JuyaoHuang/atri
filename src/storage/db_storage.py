"""Placeholder for DatabaseChatStorage (Phase 7)."""

from src.storage.interface import ChatStorageInterface


class DatabaseChatStorage(ChatStorageInterface):
    """Database-backed chat storage (reserved for Phase 7)."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    async def create_chat(self, user_id: str, character_id: str, title: str) -> dict:
        raise NotImplementedError("Database storage is reserved for Phase 7")

    async def list_chats(self, user_id: str, character_id: str | None = None) -> list[dict]:
        raise NotImplementedError("Database storage is reserved for Phase 7")

    async def get_chat(self, chat_id: str) -> dict | None:
        raise NotImplementedError("Database storage is reserved for Phase 7")

    async def update_chat(self, chat_id: str, **kwargs: str) -> dict:
        raise NotImplementedError("Database storage is reserved for Phase 7")

    async def delete_chat(self, chat_id: str) -> None:
        raise NotImplementedError("Database storage is reserved for Phase 7")

    async def append_message(
        self, chat_id: str, role: str, content: str, name: str | None = None
    ) -> dict:
        raise NotImplementedError("Database storage is reserved for Phase 7")

    async def get_messages(self, chat_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        raise NotImplementedError("Database storage is reserved for Phase 7")
