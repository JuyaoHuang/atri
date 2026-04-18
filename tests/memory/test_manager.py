"""Tests for src/memory/manager.py -- round-driven trigger scheduling.

Covers PRD US-MEM-005 acceptance criteria:
  * 25 rounds -> no L3 trigger
  * exactly 26 rounds -> L3 fires, one block appended
  * 52 rounds -> L3 fires twice, two blocks
  * 4 blocks accumulated -> L4 fires once, meta_blocks grows, active_blocks resets
  * invalid round (ai content starts with 'Error') does not increment total
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.manager import MemoryManager, _is_valid_round

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_config() -> dict[str, Any]:
    return {
        "short_term": {
            "snip": {
                "filler_words": [],
                "similarity_threshold": 0.95,
                "max_single_message_tokens": 800,
            },
            "collapse": {
                "trigger_rounds": 26,
                "compress_rounds": 20,
                "keep_recent_rounds": 6,
            },
            "super_compact": {"trigger_blocks": 4},
            "compressor": {
                "l3_role": "l3_compress",
                "l4_role": "l4_compact",
            },
        },
        "storage": {"characters_dir": "./data/characters"},
    }


def _make_llm(response: str = "mock summary") -> MagicMock:
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=response)
    return llm


def _make_factory(shared_llm: MagicMock | None = None):
    shared = shared_llm or _make_llm()

    def _factory(role: str) -> MagicMock:
        return shared

    return _factory


def _human(content: str = "hello") -> dict[str, Any]:
    return {"role": "human", "content": content, "name": "user"}


def _ai(content: str = "hi there") -> dict[str, Any]:
    return {"role": "ai", "content": content, "name": "atri"}


def _error_ai() -> dict[str, Any]:
    return {"role": "ai", "content": "Error calling endpoint", "name": "atri"}


def _new_manager(tmp_path: Path, factory=None) -> MemoryManager:
    return MemoryManager(
        _default_config(),
        factory or _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# _is_valid_round helper
# ---------------------------------------------------------------------------


def test_is_valid_round_accepts_normal_ai() -> None:
    assert _is_valid_round({"role": "ai", "content": "hi"}) is True


def test_is_valid_round_rejects_non_ai_role() -> None:
    assert _is_valid_round({"role": "human", "content": "hi"}) is False


def test_is_valid_round_rejects_empty_content() -> None:
    assert _is_valid_round({"role": "ai", "content": ""}) is False


def test_is_valid_round_rejects_error_prefix() -> None:
    assert _is_valid_round({"role": "ai", "content": "Error: upstream failed"}) is False


def test_is_valid_round_rejects_missing_content() -> None:
    assert _is_valid_round({"role": "ai"}) is False


# ---------------------------------------------------------------------------
# Trigger scheduling (PRD acceptance criteria)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_25_rounds_does_not_trigger_l3(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    for _ in range(25):
        await mgr.on_round_complete(_human(), _ai())
    assert mgr.state["total_rounds"] == 25
    assert mgr.state["active_blocks"] == []
    assert mgr.state["meta_blocks"] == []
    # All 25 rounds * 2 = 50 messages stay in recent_messages.
    assert len(mgr.state["recent_messages"]) == 50


@pytest.mark.asyncio
async def test_exactly_26_rounds_triggers_l3_once(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    for _ in range(26):
        await mgr.on_round_complete(_human(), _ai())
    assert mgr.state["total_rounds"] == 26
    assert len(mgr.state["active_blocks"]) == 1
    assert mgr.state["active_blocks"][0]["covers_rounds"] == [1, 20]
    # 20 rounds compressed -> 40 head msgs popped; 6 rounds remain = 12 msgs.
    assert len(mgr.state["recent_messages"]) == 12


@pytest.mark.asyncio
async def test_52_rounds_triggers_l3_twice(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    for _ in range(52):
        await mgr.on_round_complete(_human(), _ai())
    assert mgr.state["total_rounds"] == 52
    assert len(mgr.state["active_blocks"]) == 2
    assert mgr.state["active_blocks"][0]["covers_rounds"] == [1, 20]
    assert mgr.state["active_blocks"][1]["covers_rounds"] == [21, 40]
    # 12 rounds remain = 24 msgs (52 total - 40 compressed).
    assert len(mgr.state["recent_messages"]) == 24


@pytest.mark.asyncio
async def test_four_blocks_trigger_l4(tmp_path: Path) -> None:
    """4 L3 blocks accumulated -> L4 fires, consumes all 4."""
    mgr = _new_manager(tmp_path)
    # 104 rounds = four L3 triggers at rounds 26, 52, 78, 104.
    for _ in range(104):
        await mgr.on_round_complete(_human(), _ai())
    assert mgr.state["total_rounds"] == 104
    # After 4 L3 triggers the active_blocks hit 4, L4 consumed them.
    assert mgr.state["active_blocks"] == []
    assert len(mgr.state["meta_blocks"]) == 1
    meta = mgr.state["meta_blocks"][0]
    assert re.fullmatch(r"meta_[0-9a-f]{8}", meta["block_id"])
    assert meta["covers_rounds"] == [1, 80]
    assert len(meta["source_blocks"]) == 4


@pytest.mark.asyncio
async def test_invalid_round_does_not_increment_total(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    for _ in range(10):
        await mgr.on_round_complete(_human(), _ai())
    # 5 'Error' rounds should NOT bump total_rounds.
    for _ in range(5):
        await mgr.on_round_complete(_human(), _error_ai())
    assert mgr.state["total_rounds"] == 10
    # recent_messages only reflects the 10 valid rounds.
    assert len(mgr.state["recent_messages"]) == 20


# ---------------------------------------------------------------------------
# Persistence + LLM factory wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_term_persisted_on_each_round(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    await mgr.on_round_complete(_human(), _ai())
    file_path = tmp_path / "short_term_memory.json"
    assert file_path.exists()
    with file_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["total_rounds"] == 1
    assert payload["character"] == "atri"


@pytest.mark.asyncio
async def test_chat_history_appends_both_turns(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    await mgr.on_round_complete(_human("hi"), _ai("hello"))
    sessions_dir = tmp_path / "sessions"
    files = list(sessions_dir.glob("*.json"))
    assert len(files) == 1
    with files[0].open(encoding="utf-8") as f:
        data = json.load(f)
    # ensure_metadata is called during bootstrap so we expect:
    # [metadata, human, ai]
    assert data[0]["role"] == "metadata"
    assert data[1]["role"] == "human"
    assert data[1]["content"] == "hi"
    assert data[2]["role"] == "ai"
    assert data[2]["content"] == "hello"


@pytest.mark.asyncio
async def test_invalid_round_still_recorded_in_chat_history(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    await mgr.on_round_complete(_human("what"), _error_ai())
    sessions_dir = tmp_path / "sessions"
    files = list(sessions_dir.glob("*.json"))
    with files[0].open(encoding="utf-8") as f:
        data = json.load(f)
    # Error replies are preserved in chat_history (frontend visibility).
    assert data[-1]["role"] == "ai"
    assert data[-1]["content"].startswith("Error")


@pytest.mark.asyncio
async def test_llm_factory_called_with_configured_roles(tmp_path: Path) -> None:
    seen: list[str] = []
    shared = _make_llm("summary")

    def factory(role: str) -> MagicMock:
        seen.append(role)
        return shared

    mgr = _new_manager(tmp_path, factory=factory)
    # Up to round 26 -> triggers L3 once
    for _ in range(26):
        await mgr.on_round_complete(_human(), _ai())
    assert seen.count("l3_compress") == 1
    assert "l4_compact" not in seen


@pytest.mark.asyncio
async def test_session_id_bootstrap_follows_pattern(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    sid = mgr.active_session_id
    assert sid is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_[0-9a-f]{8}", sid)


# ---------------------------------------------------------------------------
# Long-term memory integration (US-MEM-006)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l3_trigger_calls_long_term_add(tmp_path: Path) -> None:
    """When LongTermMemory is injected, L3 should forward the raw window to mem0."""
    long_term = MagicMock()
    long_term.add = AsyncMock(return_value=None)
    long_term.search = AsyncMock(return_value=[])

    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    for _ in range(26):
        await mgr.on_round_complete(_human(), _ai())

    # L3 fired exactly once -> long_term.add awaited exactly once.
    long_term.add.assert_awaited_once()
    call_args = long_term.add.call_args
    sent_messages = call_args.args[0]
    # 20 compressed rounds = 40 raw messages.
    assert len(sent_messages) == 40
    assert call_args.kwargs["user_id"] == "alice"
    assert call_args.kwargs["agent_id"] == "atri"
    # run_id = active session_id (bootstrap-generated, matches pattern).
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_[0-9a-f]{8}", call_args.kwargs["run_id"])


@pytest.mark.asyncio
async def test_no_long_term_injection_skips_mem0_call(tmp_path: Path) -> None:
    """Default path without LongTermMemory stays functional (no error)."""
    mgr = _new_manager(tmp_path)  # long_term defaults to None
    for _ in range(26):
        await mgr.on_round_complete(_human(), _ai())
    assert mgr.state["total_rounds"] == 26
    assert len(mgr.state["active_blocks"]) == 1


@pytest.mark.asyncio
async def test_search_long_term_delegates_to_injected_backend(tmp_path: Path) -> None:
    long_term = MagicMock()
    long_term.search = AsyncMock(return_value=[{"memory": "fact1", "score": 0.9}])

    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    results = await mgr.search_long_term("hello", limit=3)
    long_term.search.assert_awaited_once_with("hello", user_id="alice", agent_id="atri", limit=3)
    assert results == [{"memory": "fact1", "score": 0.9}]


@pytest.mark.asyncio
async def test_search_long_term_returns_empty_when_no_backend(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)  # no long_term injected
    results = await mgr.search_long_term("anything")
    assert results == []
