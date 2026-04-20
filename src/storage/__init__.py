"""
Chat storage layer for Phase 5 server.

Provides abstract interface and factory for chat persistence.
"""

from src.storage.factory import create_chat_storage
from src.storage.interface import ChatStorageInterface

__all__ = ["ChatStorageInterface", "create_chat_storage"]
