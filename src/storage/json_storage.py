"""Placeholder for JSONChatStorage implementation (US-SRV-002)."""

from src.storage.interface import ChatStorageInterface


class JSONChatStorage(ChatStorageInterface):
    """JSON file-based chat storage (to be implemented in US-SRV-002)."""

    def __init__(self, base_path: str):
        self.base_path = base_path

    async def create_chat(self, user_id: str, character_id: str, title: str) -> dict:
        raise NotImplementedError("US-SRV-002")

    async def list_chats(self, user_id: str, character_id: str | None = None) -> list[dict]:
        raise NotImplementedError("US-SRV-002")

    async def get_chat(self, chat_id: str) -> dict | None:
        raise NotImplementedError("US-SRV-002")

    async def update_chat(self, chat_id: str, **kwargs: str) -> dict:
        raise NotImplementedError("US-SRV-002")

    async def delete_chat(self, chat_id: str) -> None:
        raise NotImplementedError("US-SRV-002")

    async def append_message(
        self, chat_id: str, role: str, content: str, name: str | None = None
    ) -> dict:
        raise NotImplementedError("US-SRV-002")

    async def get_messages(self, chat_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        raise NotImplementedError("US-SRV-002")
