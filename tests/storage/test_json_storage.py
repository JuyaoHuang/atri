"""Tests for JSONChatStorage."""

import pytest

from src.storage.json_storage import JSONChatStorage


@pytest.mark.asyncio
async def test_create_chat_returns_correct_structure(tmp_path):
    """Test create_chat returns metadata with all required fields."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test Chat")

    assert "id" in chat
    assert chat["title"] == "Test Chat"
    assert chat["character_id"] == "atri"
    assert "created_at" in chat
    assert "updated_at" in chat
    assert chat["message_count"] == 0


@pytest.mark.asyncio
async def test_create_chat_persists_to_disk(tmp_path):
    """Test create_chat writes index.json and session file."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test Chat")

    index_path = tmp_path / "user1" / "atri" / "index.json"
    session_path = tmp_path / "user1" / "atri" / "sessions" / f"{chat['id']}.json"

    assert index_path.exists()
    assert session_path.exists()


@pytest.mark.asyncio
async def test_list_chats_empty_directory(tmp_path):
    """Test list_chats returns empty list for new user."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chats = await storage.list_chats("user1", "atri")
    assert chats == []


@pytest.mark.asyncio
async def test_list_chats_sorted_by_updated_at(tmp_path):
    """Test list_chats returns chats sorted by updated_at descending."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat1 = await storage.create_chat("user1", "atri", "First")
    chat2 = await storage.create_chat("user1", "atri", "Second")
    chat3 = await storage.create_chat("user1", "atri", "Third")

    chats = await storage.list_chats("user1", "atri")
    assert len(chats) == 3
    assert chats[0]["id"] == chat3["id"]
    assert chats[1]["id"] == chat2["id"]
    assert chats[2]["id"] == chat1["id"]


@pytest.mark.asyncio
async def test_list_chats_filters_by_character(tmp_path):
    """Test list_chats filters by character_id."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    await storage.create_chat("user1", "atri", "Atri Chat")
    await storage.create_chat("user1", "other", "Other Chat")

    chats = await storage.list_chats("user1", "atri")
    assert len(chats) == 1
    assert chats[0]["character_id"] == "atri"


@pytest.mark.asyncio
async def test_list_chats_aggregates_all_characters(tmp_path):
    """Test list_chats without character_id returns all chats."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    await storage.create_chat("user1", "atri", "Atri Chat")
    await storage.create_chat("user1", "other", "Other Chat")

    chats = await storage.list_chats("user1")
    assert len(chats) == 2


@pytest.mark.asyncio
async def test_get_chat_returns_metadata(tmp_path):
    """Test get_chat returns correct metadata."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    created = await storage.create_chat("user1", "atri", "Test")

    retrieved = await storage.get_chat(created["id"])
    assert retrieved is not None
    assert retrieved["id"] == created["id"]
    assert retrieved["title"] == "Test"


@pytest.mark.asyncio
async def test_get_chat_returns_none_for_nonexistent(tmp_path):
    """Test get_chat returns None for nonexistent chat_id."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    result = await storage.get_chat("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_chat_modifies_title(tmp_path):
    """Test update_chat changes title and updates updated_at."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Old Title")
    original_updated_at = chat["updated_at"]

    updated = await storage.update_chat(chat["id"], title="New Title")
    assert updated["title"] == "New Title"
    assert updated["updated_at"] > original_updated_at


@pytest.mark.asyncio
async def test_update_chat_raises_on_nonexistent(tmp_path):
    """Test update_chat raises ValueError for nonexistent chat."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        await storage.update_chat("nonexistent", title="New")


@pytest.mark.asyncio
async def test_delete_chat_removes_from_index(tmp_path):
    """Test delete_chat removes chat from index."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    await storage.delete_chat(chat["id"])
    chats = await storage.list_chats("user1", "atri")
    assert len(chats) == 0


@pytest.mark.asyncio
async def test_delete_chat_removes_session_file(tmp_path):
    """Test delete_chat deletes session file."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")
    session_path = tmp_path / "user1" / "atri" / "sessions" / f"{chat['id']}.json"

    assert session_path.exists()
    await storage.delete_chat(chat["id"])
    assert not session_path.exists()


@pytest.mark.asyncio
async def test_delete_chat_raises_on_nonexistent(tmp_path):
    """Test delete_chat raises ValueError for nonexistent chat."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        await storage.delete_chat("nonexistent")


@pytest.mark.asyncio
async def test_append_message_adds_to_session(tmp_path):
    """Test append_message adds message to session file."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    msg = await storage.append_message(chat["id"], "human", "Hello")
    assert msg["role"] == "human"
    assert msg["content"] == "Hello"
    assert "timestamp" in msg


@pytest.mark.asyncio
async def test_append_message_updates_message_count(tmp_path):
    """Test append_message increments message_count in index."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    await storage.append_message(chat["id"], "human", "Hello")
    await storage.append_message(chat["id"], "ai", "Hi")

    updated = await storage.get_chat(chat["id"])
    assert updated["message_count"] == 2


@pytest.mark.asyncio
async def test_append_message_updates_updated_at(tmp_path):
    """Test append_message updates updated_at timestamp."""
    import asyncio

    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")
    original_updated_at = chat["updated_at"]

    await asyncio.sleep(0.01)  # Ensure timestamp difference
    await storage.append_message(chat["id"], "human", "Hello")
    updated = await storage.get_chat(chat["id"])
    assert updated["updated_at"] > original_updated_at


@pytest.mark.asyncio
async def test_append_message_with_name(tmp_path):
    """Test append_message includes name field when provided."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    msg = await storage.append_message(chat["id"], "human", "Hello", name="Alice")
    assert msg["name"] == "Alice"


@pytest.mark.asyncio
async def test_append_message_raises_on_nonexistent(tmp_path):
    """Test append_message raises ValueError for nonexistent chat."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        await storage.append_message("nonexistent", "human", "Hello")


@pytest.mark.asyncio
async def test_get_messages_returns_all_messages(tmp_path):
    """Test get_messages returns all messages."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    await storage.append_message(chat["id"], "human", "Hello")
    await storage.append_message(chat["id"], "ai", "Hi")

    messages = await storage.get_messages(chat["id"])
    assert len(messages) == 2
    assert messages[0]["content"] == "Hello"
    assert messages[1]["content"] == "Hi"


@pytest.mark.asyncio
async def test_get_messages_pagination_limit(tmp_path):
    """Test get_messages respects limit parameter."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    for i in range(5):
        await storage.append_message(chat["id"], "human", f"Message {i}")

    messages = await storage.get_messages(chat["id"], limit=3)
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_get_messages_pagination_offset(tmp_path):
    """Test get_messages respects offset parameter."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    for i in range(5):
        await storage.append_message(chat["id"], "human", f"Message {i}")

    messages = await storage.get_messages(chat["id"], limit=2, offset=2)
    assert len(messages) == 2
    assert messages[0]["content"] == "Message 2"
    assert messages[1]["content"] == "Message 3"


@pytest.mark.asyncio
async def test_get_messages_raises_on_nonexistent(tmp_path):
    """Test get_messages raises ValueError for nonexistent chat."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        await storage.get_messages("nonexistent")


@pytest.mark.asyncio
async def test_chat_id_format(tmp_path):
    """Test chat ID follows YYYYMMDD_uuid8 format."""
    storage = JSONChatStorage(base_path=str(tmp_path))
    chat = await storage.create_chat("user1", "atri", "Test")

    import re

    assert re.match(r"^\d{8}_[a-f0-9]{8}$", chat["id"])
