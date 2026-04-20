"""
Abstract interface for chat storage.

Defines the contract for chat persistence implementations (JSON, database, etc.).
"""

from abc import ABC, abstractmethod


class ChatStorageInterface(ABC):
    """Abstract interface for chat storage operations."""

    @abstractmethod
    async def create_chat(self, user_id: str, character_id: str, title: str) -> dict:
        """
        Create a new chat session.

        Args:
            user_id: User identifier
            character_id: Character identifier
            title: Chat title (LLM-generated or fallback)

        Returns:
            Chat metadata dict with keys: id, title, character_id,
            created_at, updated_at, message_count
        """
        ...

    @abstractmethod
    async def list_chats(self, user_id: str, character_id: str | None = None) -> list[dict]:
        """
        List user's chat sessions, sorted by updated_at descending.

        Args:
            user_id: User identifier
            character_id: Optional filter by character

        Returns:
            List of chat metadata dicts
        """
        ...

    @abstractmethod
    async def get_chat(self, chat_id: str) -> dict | None:
        """
        Get chat metadata by ID.

        Args:
            chat_id: Chat identifier

        Returns:
            Chat metadata dict or None if not found
        """
        ...

    @abstractmethod
    async def update_chat(self, chat_id: str, **kwargs: str) -> dict:
        """
        Update chat metadata (e.g., title).

        Args:
            chat_id: Chat identifier
            **kwargs: Fields to update (e.g., title="New Title")

        Returns:
            Updated chat metadata dict

        Raises:
            ValueError: If chat_id not found
        """
        ...

    @abstractmethod
    async def delete_chat(self, chat_id: str) -> None:
        """
        Delete a chat session and its messages.

        Args:
            chat_id: Chat identifier

        Raises:
            ValueError: If chat_id not found
        """
        ...

    @abstractmethod
    async def append_message(
        self, chat_id: str, role: str, content: str, name: str | None = None
    ) -> dict:
        """
        Append a message to chat history.

        Args:
            chat_id: Chat identifier
            role: Message role (human/ai/system)
            content: Message content
            name: Optional speaker name

        Returns:
            Message dict with keys: role, content, name, timestamp

        Raises:
            ValueError: If chat_id not found
        """
        ...

    @abstractmethod
    async def get_messages(self, chat_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """
        Get chat message history with pagination.

        Args:
            chat_id: Chat identifier
            limit: Maximum messages to return
            offset: Number of messages to skip

        Returns:
            List of message dicts

        Raises:
            ValueError: If chat_id not found
        """
        ...
