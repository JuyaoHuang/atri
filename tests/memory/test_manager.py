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

from src.memory.chat_history import ChatHistoryWriter
from src.memory.manager import MemoryManager, _is_valid_round
from src.memory.short_term import ShortTermStore

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


# ---------------------------------------------------------------------------
# Session lifecycle (US-MEM-007)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_session_generates_matching_id_format(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    sid = await mgr.start_session()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_[0-9a-f]{8}", sid)
    assert mgr.active_session_id == sid


@pytest.mark.asyncio
async def test_start_session_writes_single_metadata_row(tmp_path: Path) -> None:
    mgr = _new_manager(tmp_path)
    sid = await mgr.start_session()
    session_file = tmp_path / "sessions" / f"{sid}.json"
    assert session_file.exists()
    with session_file.open(encoding="utf-8") as f:
        data = json.load(f)
    metadata_rows = [row for row in data if row.get("role") == "metadata"]
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["session_id"] == sid
    assert metadata_rows[0]["character"] == "atri"


@pytest.mark.asyncio
async def test_close_session_with_pending_messages_pushes_long_term(tmp_path: Path) -> None:
    long_term = MagicMock()
    long_term.add = AsyncMock(return_value=None)
    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    await mgr.start_session()
    for _ in range(5):
        await mgr.on_round_complete(_human(), _ai())
    await mgr.close_session()

    long_term.add.assert_awaited_once()
    call = long_term.add.call_args
    assert len(call.args[0]) == 10  # 5 rounds * 2 messages
    assert call.kwargs["user_id"] == "alice"
    assert call.kwargs["agent_id"] == "atri"
    assert mgr.active_session_id is None


@pytest.mark.asyncio
async def test_close_session_with_no_new_messages_skips_long_term(tmp_path: Path) -> None:
    """start -> close with zero rounds must not touch mem0 (idempotency)."""
    long_term = MagicMock()
    long_term.add = AsyncMock(return_value=None)
    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    await mgr.start_session()
    await mgr.close_session()

    long_term.add.assert_not_awaited()
    assert mgr.active_session_id is None


@pytest.mark.asyncio
async def test_close_session_is_idempotent_across_double_calls(tmp_path: Path) -> None:
    long_term = MagicMock()
    long_term.add = AsyncMock(return_value=None)
    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    await mgr.start_session()
    await mgr.on_round_complete(_human(), _ai())
    await mgr.close_session()
    # Calling close again with no active session is a silent no-op.
    await mgr.close_session()
    assert long_term.add.await_count == 1


# ---------------------------------------------------------------------------
# Session resume (US-MEM-008)
# ---------------------------------------------------------------------------


def _seed_session_files(
    tmp_path: Path,
    session_id: str,
    character: str,
    *,
    stored_rounds: int,
    chat_rounds: int,
    short_term_body: str | None = None,
) -> None:
    """Create synthetic short_term_memory.json + sessions/{id}.json for resume tests.

    Writes ``stored_rounds`` rounds into short_term (with matching
    recent_messages) and ``chat_rounds`` valid (human, ai) pairs into
    chat_history. Passing ``short_term_body`` overrides the short-term
    payload verbatim (used to inject invalid JSON for corruption tests).
    """
    hist = ChatHistoryWriter(tmp_path, session_id, character)
    hist.ensure_metadata()
    for i in range(chat_rounds):
        hist.append_human(f"h{i}", name="alice")
        hist.append_ai(f"a{i}", name=character)

    short = ShortTermStore(tmp_path, session_id, character)
    if short_term_body is not None:
        short.path.write_text(short_term_body, encoding="utf-8")
        return
    state = ShortTermStore.get_skeleton(session_id, character)
    state["total_rounds"] = stored_rounds
    for i in range(stored_rounds):
        state["recent_messages"].append({"role": "human", "content": f"h{i}"})
        state["recent_messages"].append({"role": "ai", "content": f"a{i}"})
    short.save(state)


def _make_manager_with_mock_long_term(
    tmp_path: Path,
) -> tuple[MemoryManager, MagicMock]:
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
    return mgr, long_term


@pytest.mark.asyncio
async def test_resume_consistent_preserves_state(tmp_path: Path) -> None:
    session_id = "2026-04-19_aabbccdd"
    _seed_session_files(tmp_path, session_id, "atri", stored_rounds=10, chat_rounds=10)
    mgr, long_term = _make_manager_with_mock_long_term(tmp_path)

    await mgr.resume_session(session_id)

    assert mgr.active_session_id == session_id
    assert mgr.state["total_rounds"] == 10
    assert len(mgr.state["recent_messages"]) == 20
    long_term.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_behind_catches_up_tail(tmp_path: Path) -> None:
    """Stored total_rounds=10, chat_history=12 valid rounds -> replay 2 missing."""
    session_id = "2026-04-19_1122aabb"
    _seed_session_files(tmp_path, session_id, "atri", stored_rounds=10, chat_rounds=12)
    mgr, long_term = _make_manager_with_mock_long_term(tmp_path)

    await mgr.resume_session(session_id)

    assert mgr.state["total_rounds"] == 12
    # 12 rounds * 2 messages = 24 total in recent_messages.
    assert len(mgr.state["recent_messages"]) == 24
    # Trailing content comes from the appended chat_history pairs.
    assert mgr.state["recent_messages"][-2]["content"] == "h11"
    assert mgr.state["recent_messages"][-1]["content"] == "a11"
    # No L3 boundary crossed (12 < 26), no long_term.add invoked.
    long_term.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_on_boundary_triggers_l3(tmp_path: Path) -> None:
    """Stored total_rounds=20, chat_history=26 -> replay lands on 26 => L3 fires."""
    session_id = "2026-04-19_c0ffee01"
    _seed_session_files(tmp_path, session_id, "atri", stored_rounds=20, chat_rounds=26)
    mgr, long_term = _make_manager_with_mock_long_term(tmp_path)

    await mgr.resume_session(session_id)

    assert mgr.state["total_rounds"] == 26
    assert len(mgr.state["active_blocks"]) == 1
    assert mgr.state["active_blocks"][0]["covers_rounds"] == [1, 20]
    # L3 trigger -> long_term.add invoked once with the 40-msg compress window.
    long_term.add.assert_awaited_once()
    assert len(long_term.add.call_args.args[0]) == 40
    # 6 rounds remain raw = 12 msgs.
    assert len(mgr.state["recent_messages"]) == 12


@pytest.mark.asyncio
async def test_resume_corrupt_json_rebuilds_from_chat_history(tmp_path: Path) -> None:
    """Broken short_term.json -> full rebuild replays every chat_history round."""
    session_id = "2026-04-19_baddada1"
    _seed_session_files(
        tmp_path,
        session_id,
        "atri",
        stored_rounds=0,
        chat_rounds=26,
        short_term_body='{"this is not": valid json',
    )
    mgr, long_term = _make_manager_with_mock_long_term(tmp_path)

    await mgr.resume_session(session_id)

    assert mgr.state["total_rounds"] == 26
    assert len(mgr.state["active_blocks"]) == 1
    assert mgr.state["active_blocks"][0]["covers_rounds"] == [1, 20]
    # Rebuild replays L3 windows through long_term.add for idempotency.
    long_term.add.assert_awaited_once()
    assert len(mgr.state["recent_messages"]) == 12


@pytest.mark.asyncio
async def test_resume_chat_history_with_trailing_garbage(tmp_path: Path) -> None:
    """chat_history has a trailing malformed record -> tolerant parse recovers prefix."""
    session_id = "2026-04-19_deadbeef"
    _seed_session_files(tmp_path, session_id, "atri", stored_rounds=5, chat_rounds=5)
    # Append trailing garbage to chat_history (simulating a partial write crash).
    hist_path = tmp_path / "sessions" / f"{session_id}.json"
    original = hist_path.read_text(encoding="utf-8")
    # Transform ``[..., {last}]`` -> ``[..., {last}, {corrupt...``
    corrupted = original.rstrip()[:-1] + ', {"role": "ai", "content": "partial'
    hist_path.write_text(corrupted, encoding="utf-8")

    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    # Must NOT raise; tolerant parse yields the 5 well-formed rounds.
    await mgr.resume_session(session_id)
    assert mgr.state["total_rounds"] == 5


# ---------------------------------------------------------------------------
# LLM context builder (US-MEM-009, §3.5 payload order)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_llm_context_full_order(tmp_path: Path) -> None:
    """system -> long_term -> meta -> active -> recent -> user, in that order."""
    long_term = MagicMock()
    long_term.search = AsyncMock(
        return_value=[
            {"memory": "likes bubble tea", "score": 0.9},
            {"memory": "plays piano", "score": 0.85},
        ]
    )
    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    # Inject blocks + recent messages directly into state.
    mgr.state["meta_blocks"] = [
        {"summary": "meta_newer", "covers_rounds": [21, 80]},
        {"summary": "meta_older", "covers_rounds": [1, 20]},
    ]
    mgr.state["active_blocks"] = [
        {"summary": "active_1", "covers_rounds": [81, 100]},
        {"summary": "active_2", "covers_rounds": [101, 120]},
    ]
    mgr.state["recent_messages"] = [
        {"role": "human", "content": "h1"},
        {"role": "ai", "content": "a1"},
    ]

    messages = await mgr.build_llm_context("what's up", system_prompt="You are atri")

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    assert roles == [
        "system",  # system_prompt
        "system",  # long-term wrapper
        "system",  # meta_older (reversed)
        "system",  # meta_newer
        "system",  # active_1
        "system",  # active_2
        "user",  # h1
        "assistant",  # a1
        "user",  # final user_input
    ]
    assert contents[0] == "You are atri"
    assert contents[1].startswith("关于这位用户，你记得：")
    assert "- likes bubble tea" in contents[1]
    assert "- plays piano" in contents[1]
    assert contents[2] == "meta_older"
    assert contents[3] == "meta_newer"
    assert contents[4] == "active_1"
    assert contents[5] == "active_2"
    assert contents[6] == "h1"
    assert contents[7] == "a1"
    assert contents[8] == "what's up"


@pytest.mark.asyncio
async def test_build_llm_context_empty_system_prompt_omits_position_1(
    tmp_path: Path,
) -> None:
    long_term = MagicMock()
    long_term.search = AsyncMock(return_value=[{"memory": "likes cats", "score": 0.9}])
    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    messages = await mgr.build_llm_context("hi", system_prompt="")
    # First message is the long-term wrapper (no preceding system_prompt).
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("关于这位用户，你记得：")
    assert messages[-1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_build_llm_context_empty_long_term_omits_position_2(
    tmp_path: Path,
) -> None:
    """When search returns [], the long-term wrapper message is skipped."""
    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    messages = await mgr.build_llm_context("hello", system_prompt="sysprompt")
    # With no blocks / recent messages, layout is just [sysprompt, user].
    assert messages == [
        {"role": "system", "content": "sysprompt"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.asyncio
async def test_build_llm_context_role_mapping(tmp_path: Path) -> None:
    """human -> user, ai -> assistant, system -> system."""
    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    mgr.state["recent_messages"] = [
        {"role": "human", "content": "q1"},
        {"role": "ai", "content": "r1"},
        {"role": "system", "content": "[interrupt]"},
    ]
    messages = await mgr.build_llm_context("next")
    # Drop the final user_input so we inspect only the mapped tail.
    mapped = messages[:-1]
    assert mapped[-3] == {"role": "user", "content": "q1"}
    assert mapped[-2] == {"role": "assistant", "content": "r1"}
    assert mapped[-1] == {"role": "system", "content": "[interrupt]"}


@pytest.mark.asyncio
async def test_build_llm_context_long_term_bullet_format(tmp_path: Path) -> None:
    long_term = MagicMock()
    long_term.search = AsyncMock(
        return_value=[
            {"memory": "fact A", "score": 0.9},
            {"memory": "fact B", "score": 0.85},
            {"memory": "fact C", "score": 0.8},
        ]
    )
    mgr = MemoryManager(
        _default_config(),
        _make_factory(),
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
        long_term=long_term,
    )
    messages = await mgr.build_llm_context("q")
    wrapper = messages[0]  # no system_prompt -> long-term wrapper is first
    assert wrapper["content"] == ("关于这位用户，你记得：\n- fact A\n- fact B\n- fact C")


@pytest.mark.asyncio
async def test_build_llm_context_unknown_role_raises(tmp_path: Path) -> None:
    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    mgr.state["recent_messages"] = [{"role": "spaceship", "content": "??"}]
    with pytest.raises(ValueError, match="Unknown role"):
        await mgr.build_llm_context("q")


@pytest.mark.asyncio
async def test_build_llm_context_does_not_call_llm(tmp_path: Path) -> None:
    """The builder must only assemble; LLM call is ChatAgent's job."""
    shared_llm = _make_llm()
    factory_calls: list[str] = []

    def _factory(role: str) -> MagicMock:
        factory_calls.append(role)
        return shared_llm

    mgr = MemoryManager(
        _default_config(),
        _factory,
        character="atri",
        user_id="alice",
        character_dir=tmp_path,
    )
    await mgr.build_llm_context("hello", system_prompt="s")
    shared_llm.chat_completion.assert_not_awaited()
    assert factory_calls == []
