"""Tests for storage interface and factory."""

import pytest

from src.storage.factory import create_chat_storage
from src.storage.interface import ChatStorageInterface
from src.storage.json_storage import JSONChatStorage


def test_interface_cannot_be_instantiated():
    """Test ChatStorageInterface is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        ChatStorageInterface()  # type: ignore


def test_factory_creates_json_storage(tmp_path):
    """Test factory creates JSONChatStorage when mode is 'json'."""
    config = {
        "storage": {
            "mode": "json",
            "json": {"base_path": str(tmp_path / "chats")},
        }
    }
    storage = create_chat_storage(config)
    assert isinstance(storage, JSONChatStorage)
    assert isinstance(storage, ChatStorageInterface)


def test_factory_raises_on_database_mode():
    """Test factory raises NotImplementedError for database mode (Phase 7)."""
    config = {
        "storage": {
            "mode": "database",
            "database": {"url": "postgresql://localhost/test"},
        }
    }
    with pytest.raises(NotImplementedError, match="Phase 7"):
        create_chat_storage(config)


def test_factory_raises_on_unknown_mode():
    """Test factory raises ValueError for unknown storage mode."""
    config = {"storage": {"mode": "invalid"}}
    with pytest.raises(ValueError, match="Unknown storage mode"):
        create_chat_storage(config)


def test_factory_defaults_to_json():
    """Test factory defaults to json mode when mode is not specified."""
    config = {"storage": {"json": {"base_path": "data/chats"}}}
    storage = create_chat_storage(config)
    assert isinstance(storage, JSONChatStorage)
