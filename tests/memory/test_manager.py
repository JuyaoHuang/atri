"""Tests for src/memory/manager.py -- round-driven trigger scheduling.

Covers PRD US-MEM-005 acceptance criteria:
  * 25 rounds -> no L3 trigger
  * exactly 26 rounds -> L3 fires, one block appended
  * 52 rounds -> L3 fires twice, two blocks
  * 4 blocks accumulated -> L4 fires once, meta_blocks grows, active_blocks resets
  * invalid round (ai content starts with 'Error') does not increment total

针对 src/memory/manager.py 的测试——按轮次驱动的触发调度。

覆盖 PRD US-MEM-005 的验收标准：
  * 25 轮 -> 不触发 L3
  * 恰好 26 轮 -> L3 触发一次，追加一个块
  * 52 轮 -> L3 触发两次，两个块
  * 累计 4 个块 -> L4 触发一次，meta_blocks 增长，active_blocks 重置
  * 无效轮次（ai 内容以 'Error' 开头）不增加 total_rounds

此外还涵盖：短期状态每轮持久化、chat_history 同时追加两端消息、错误回复
仍保留于 chat_history（供前端可视）、LLM 工厂按配置角色调用、session_id
形态为 ``{YYYY-MM-DD}_{8-hex}``；长期记忆集成（US-MEM-006）：L3 触发时
转发原始窗口到 mem0、无 LongTermMemory 注入时跳过、search_long_term 的
委托与空返回；会话生命周期（US-MEM-007）：start_session 生成 id 与写
metadata 行、close_session 推送未压缩尾巴并幂等；会话恢复（US-MEM-008）：
一致 / 落后追赶 / 触发边界引发 L3 / 损坏的 JSON 全量重建 / chat_history
尾部容错；以及 LLM 上下文构建（US-MEM-009，§3.5 载荷顺序）：system ->
long_term -> meta -> active -> recent -> user、空占位省略、角色映射
（human->user / ai->assistant / system->system）、长期事实的要点格式、
未知角色抛错、构建不触发 LLM 调用。
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
# 辅助函数
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
# _is_valid_round 辅助函数
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
# 触发调度（PRD 验收标准）
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
    # 全部 25 轮 * 2 = 50 条消息保留在 recent_messages 中。
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
    # 压缩 20 轮 -> 弹出头部 40 条消息；余下 6 轮 = 12 条消息。
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
    # 余下 12 轮 = 24 条消息（总计 52 - 压缩 40）。
    assert len(mgr.state["recent_messages"]) == 24


@pytest.mark.asyncio
async def test_four_blocks_trigger_l4(tmp_path: Path) -> None:
    """4 L3 blocks accumulated -> L4 fires, consumes all 4.

    累计 4 个 L3 块 -> L4 触发，消耗全部 4 个块。
    """
    mgr = _new_manager(tmp_path)
    # 104 rounds = four L3 triggers at rounds 26, 52, 78, 104.
    # 104 轮 = 分别在第 26、52、78、104 轮触发 4 次 L3。
    for _ in range(104):
        await mgr.on_round_complete(_human(), _ai())
    assert mgr.state["total_rounds"] == 104
    # After 4 L3 triggers the active_blocks hit 4, L4 consumed them.
    # 4 次 L3 触发后 active_blocks 达到 4，L4 将其全部消耗。
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
    # 5 次 'Error' 轮次不应使 total_rounds 自增。
    for _ in range(5):
        await mgr.on_round_complete(_human(), _error_ai())
    assert mgr.state["total_rounds"] == 10
    # recent_messages only reflects the 10 valid rounds.
    # recent_messages 只反映 10 个有效轮次。
    assert len(mgr.state["recent_messages"]) == 20


# ---------------------------------------------------------------------------
# Persistence + LLM factory wiring
# 持久化 + LLM 工厂连线
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
    # 引导时会调用 ensure_metadata，因此期望的顺序为：
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
    # 错误回复保留在 chat_history 中（供前端可见）。
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
    # 执行到第 26 轮 -> 触发一次 L3
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
# 长期记忆集成（US-MEM-006）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l3_trigger_calls_long_term_add(tmp_path: Path) -> None:
    """When LongTermMemory is injected, L3 should forward the raw window to mem0.

    当注入 LongTermMemory 时，L3 应将原始窗口转发给 mem0。
    """
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
    # L3 恰好触发一次 -> long_term.add 恰好被等待一次。
    long_term.add.assert_awaited_once()
    call_args = long_term.add.call_args
    sent_messages = call_args.args[0]
    # 20 compressed rounds = 40 raw messages.
    # 压缩 20 轮 = 40 条原始消息。
    assert len(sent_messages) == 40
    assert call_args.kwargs["user_id"] == "alice"
    assert call_args.kwargs["agent_id"] == "atri"
    # run_id = active session_id (bootstrap-generated, matches pattern).
    # run_id = 激活的 session_id（引导时生成，符合格式）。
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_[0-9a-f]{8}", call_args.kwargs["run_id"])


@pytest.mark.asyncio
async def test_no_long_term_injection_skips_mem0_call(tmp_path: Path) -> None:
    """Default path without LongTermMemory stays functional (no error).

    未注入 LongTermMemory 的默认路径仍应正常工作（不报错）。
    """
    # long_term 默认为 None
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
    # 未注入 long_term
    mgr = _new_manager(tmp_path)  # no long_term injected
    results = await mgr.search_long_term("anything")
    assert results == []


# ---------------------------------------------------------------------------
# Session lifecycle (US-MEM-007)
# 会话生命周期（US-MEM-007）
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
    """start -> close with zero rounds must not touch mem0 (idempotency).

    start -> close（零轮次）时不得触达 mem0（幂等性）。
    """
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
    # 在无激活会话时再次调用 close 是静默的空操作。
    await mgr.close_session()
    assert long_term.add.await_count == 1


# ---------------------------------------------------------------------------
# Session resume (US-MEM-008)
# 会话恢复（US-MEM-008）
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

    为恢复测试创建合成的 short_term_memory.json + sessions/{id}.json。

    向 short_term 写入 ``stored_rounds`` 轮（并同步到 recent_messages），
    向 chat_history 写入 ``chat_rounds`` 对合法的 (human, ai) 消息。
    传入 ``short_term_body`` 会按原样覆盖短期载荷（用于注入非法 JSON
    以进行损坏测试）。
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
    """Stored total_rounds=10, chat_history=12 valid rounds -> replay 2 missing.

    已存 total_rounds=10，chat_history 有 12 个合法轮次 -> 重放缺失的 2 个。
    """
    session_id = "2026-04-19_1122aabb"
    _seed_session_files(tmp_path, session_id, "atri", stored_rounds=10, chat_rounds=12)
    mgr, long_term = _make_manager_with_mock_long_term(tmp_path)

    await mgr.resume_session(session_id)

    assert mgr.state["total_rounds"] == 12
    # 12 rounds * 2 messages = 24 total in recent_messages.
    # 12 轮 * 2 条消息 = recent_messages 总计 24 条。
    assert len(mgr.state["recent_messages"]) == 24
    # Trailing content comes from the appended chat_history pairs.
    # 尾部内容来自 chat_history 中追加的消息对。
    assert mgr.state["recent_messages"][-2]["content"] == "h11"
    assert mgr.state["recent_messages"][-1]["content"] == "a11"
    # No L3 boundary crossed (12 < 26), no long_term.add invoked.
    # 未越过 L3 边界（12 < 26），故 long_term.add 未被调用。
    long_term.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_on_boundary_triggers_l3(tmp_path: Path) -> None:
    """Stored total_rounds=20, chat_history=26 -> replay lands on 26 => L3 fires.

    已存 total_rounds=20，chat_history=26 -> 重放至 26 => L3 触发。
    """
    session_id = "2026-04-19_c0ffee01"
    _seed_session_files(tmp_path, session_id, "atri", stored_rounds=20, chat_rounds=26)
    mgr, long_term = _make_manager_with_mock_long_term(tmp_path)

    await mgr.resume_session(session_id)

    assert mgr.state["total_rounds"] == 26
    assert len(mgr.state["active_blocks"]) == 1
    assert mgr.state["active_blocks"][0]["covers_rounds"] == [1, 20]
    # L3 trigger -> long_term.add invoked once with the 40-msg compress window.
    # L3 触发 -> long_term.add 被调用一次，传入 40 条消息的压缩窗口。
    long_term.add.assert_awaited_once()
    assert len(long_term.add.call_args.args[0]) == 40
    # 6 rounds remain raw = 12 msgs.
    # 保留 6 轮原始消息 = 12 条。
    assert len(mgr.state["recent_messages"]) == 12


@pytest.mark.asyncio
async def test_resume_corrupt_json_rebuilds_from_chat_history(tmp_path: Path) -> None:
    """Broken short_term.json -> full rebuild replays every chat_history round.

    损坏的 short_term.json -> 从 chat_history 全量重建并重放所有轮次。
    """
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
    # 重建时通过 long_term.add 重放 L3 窗口以保证幂等性。
    long_term.add.assert_awaited_once()
    assert len(mgr.state["recent_messages"]) == 12


@pytest.mark.asyncio
async def test_resume_chat_history_with_trailing_garbage(tmp_path: Path) -> None:
    """chat_history has a trailing malformed record -> tolerant parse recovers prefix.

    chat_history 尾部存在畸形记录 -> 容错解析恢复可解析前缀。
    """
    session_id = "2026-04-19_deadbeef"
    _seed_session_files(tmp_path, session_id, "atri", stored_rounds=5, chat_rounds=5)
    # Append trailing garbage to chat_history (simulating a partial write crash).
    # 向 chat_history 追加尾部垃圾内容（模拟写入途中崩溃）。
    hist_path = tmp_path / "sessions" / f"{session_id}.json"
    original = hist_path.read_text(encoding="utf-8")
    # Transform ``[..., {last}]`` -> ``[..., {last}, {corrupt...``
    # 将 ``[..., {last}]`` 变为 ``[..., {last}, {corrupt...``
    corrupted = original.rstrip()[:-1] + ', {"role": "ai", "content": "partial'
    hist_path.write_text(corrupted, encoding="utf-8")

    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    # Must NOT raise; tolerant parse yields the 5 well-formed rounds.
    # 不得抛异常；容错解析返回 5 个合法轮次。
    await mgr.resume_session(session_id)
    assert mgr.state["total_rounds"] == 5


# ---------------------------------------------------------------------------
# LLM context builder (US-MEM-009, §3.5 payload order)
# LLM 上下文构建器（US-MEM-009，§3.5 载荷顺序）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_llm_context_full_order(tmp_path: Path) -> None:
    """system -> long_term -> meta -> active -> recent -> user, in that order.

    system -> long_term -> meta -> active -> recent -> user，按此顺序排列。
    """
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
    # 直接将块与最近消息注入到状态中。
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
        "system",  # long-term wrapper    长期记忆包裹消息
        "system",  # meta_older (reversed)    meta_older（倒序）
        "system",  # meta_newer
        "system",  # active_1
        "system",  # active_2
        "user",  # h1
        "assistant",  # a1
        "user",  # final user_input    最终的用户输入
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
    # 首条消息是长期记忆包裹（前面没有 system_prompt）。
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("关于这位用户，你记得：")
    assert messages[-1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_build_llm_context_empty_long_term_omits_position_2(
    tmp_path: Path,
) -> None:
    """When search returns [], the long-term wrapper message is skipped.

    当 search 返回 [] 时，长期记忆包裹消息会被跳过。
    """
    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    messages = await mgr.build_llm_context("hello", system_prompt="sysprompt")
    # With no blocks / recent messages, layout is just [sysprompt, user].
    # 没有块 / 最近消息时，布局只剩 [sysprompt, user]。
    assert messages == [
        {"role": "system", "content": "sysprompt"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.asyncio
async def test_build_llm_context_appends_runtime_datetime_context(tmp_path: Path) -> None:
    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    runtime_context = {
        "datetime": {
            "iso": "2026-04-25T04:34:00.000Z",
            "local": "2026/4/25 12:34:00",
            "time_zone": "Asia/Shanghai",
            "utc_offset": "UTC+08:00",
        }
    }

    messages = await mgr.build_llm_context("what time is it?", runtime_context=runtime_context)

    final_content = messages[-1]["content"]
    assert final_content.startswith("what time is it?\n\n<context>")
    assert (
        '<module name="system:datetime">Current datetime: '
        "2026-04-25T04:34:00.000Z "
        "(2026/4/25 12:34:00; Asia/Shanghai; UTC+08:00)</module>"
    ) in final_content
    assert final_content.endswith("</context>")


@pytest.mark.asyncio
async def test_build_llm_context_escapes_runtime_datetime_context(tmp_path: Path) -> None:
    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    runtime_context = {
        "datetime": {
            "iso": "2026-04-25T04:34:00.000Z",
            "local": "<local & browser>",
        }
    }

    messages = await mgr.build_llm_context("q", runtime_context=runtime_context)

    final_content = messages[-1]["content"]
    assert "<local & browser>" not in final_content
    assert "&lt;local &amp; browser&gt;" in final_content


@pytest.mark.asyncio
async def test_build_llm_context_role_mapping(tmp_path: Path) -> None:
    """human -> user, ai -> assistant, system -> system.

    human -> user，ai -> assistant，system -> system。
    """
    mgr, _long_term = _make_manager_with_mock_long_term(tmp_path)
    mgr.state["recent_messages"] = [
        {"role": "human", "content": "q1"},
        {"role": "ai", "content": "r1"},
        {"role": "system", "content": "[interrupt]"},
    ]
    messages = await mgr.build_llm_context("next")
    # Drop the final user_input so we inspect only the mapped tail.
    # 去掉末尾的 user_input，仅检查被映射后的尾部内容。
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
    # 无 system_prompt 时，长期记忆包裹为首条消息
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
    """The builder must only assemble; LLM call is ChatAgent's job.

    构建器只负责组装；LLM 调用是 ChatAgent 的职责。
    """
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


# ---------------------------------------------------------------------------
# append_system_note (US-AGT-002 — Phase 4 error-path support)
# append_system_note（US-AGT-002——Phase 4 错误路径支持）
# ---------------------------------------------------------------------------


def test_append_system_note_writes_system_row(tmp_path: Path) -> None:
    """A call appends exactly one role=system entry with the given content.

    一次调用在 chat_history 末尾追加一条 role=system 记录，content 等于入参。
    """
    mgr = _new_manager(tmp_path)
    assert mgr.chat_history is not None
    before = list(mgr.chat_history.iter_messages())

    mgr.append_system_note("[LLM call failed: LLMConnectionError: timeout]")

    after = list(mgr.chat_history.iter_messages())
    assert len(after) == len(before) + 1
    last = after[-1]
    assert last["role"] == "system"
    assert last["content"] == "[LLM call failed: LLMConnectionError: timeout]"


def test_append_system_note_does_not_mutate_total_rounds(tmp_path: Path) -> None:
    """Calling append_system_note leaves total_rounds untouched.

    调用 append_system_note 后 total_rounds 保持不变。
    """
    mgr = _new_manager(tmp_path)
    mgr.state["total_rounds"] = 7

    mgr.append_system_note("note A")
    mgr.append_system_note("note B")

    assert mgr.state["total_rounds"] == 7


def test_append_system_note_does_not_mutate_recent_messages(tmp_path: Path) -> None:
    """Calling append_system_note leaves recent_messages identity and contents intact.

    调用后 recent_messages 的标识和内容都保持不变。
    """
    mgr = _new_manager(tmp_path)
    seed = [{"role": "human", "content": "hi"}, {"role": "ai", "content": "hello"}]
    mgr.state["recent_messages"] = seed

    mgr.append_system_note("note")

    assert mgr.state["recent_messages"] is seed
    assert mgr.state["recent_messages"] == [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "hello"},
    ]


def test_append_system_note_does_not_trigger_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with state that would normally trip L3/L4, append_system_note must not fire them.

    即便状态已达 L3/L4 触发条件，append_system_note 也绝不触发压缩。
    """
    l3_mock = AsyncMock()
    l4_mock = AsyncMock()
    monkeypatch.setattr("src.memory.manager.l3_collapse", l3_mock)
    monkeypatch.setattr("src.memory.manager.l4_super_compact", l4_mock)

    mgr = _new_manager(tmp_path)
    # Arrange state that would trigger L3 on round-complete AND L4 on a subsequent check.
    # 构造足以触发 L3（轮次倍数）且满足 L4（active_blocks>=trigger）的状态。
    mgr.state["total_rounds"] = 26
    mgr.state["active_blocks"] = [
        {"block_id": f"b{i}", "summary": "s", "covers_rounds": [0, 0]} for i in range(4)
    ]

    mgr.append_system_note("note")

    l3_mock.assert_not_called()
    l4_mock.assert_not_called()


def test_append_system_note_multiple_calls_append_multiple_rows(tmp_path: Path) -> None:
    """Consecutive calls are not deduped; each produces a new chat_history row.

    连续调用不会去重，每次都追加新行。
    """
    mgr = _new_manager(tmp_path)
    assert mgr.chat_history is not None
    before = len(list(mgr.chat_history.iter_messages()))

    mgr.append_system_note("note A")
    mgr.append_system_note("note A")  # same content, intentionally
    mgr.append_system_note("note B")

    entries = list(mgr.chat_history.iter_messages())
    assert len(entries) == before + 3
    system_contents = [e["content"] for e in entries if e["role"] == "system"]
    assert system_contents[-3:] == ["note A", "note A", "note B"]


@pytest.mark.asyncio
async def test_append_system_note_asserts_when_no_active_session(tmp_path: Path) -> None:
    """After close_session, append_system_note raises AssertionError.

    close_session 之后调用 append_system_note 抛 AssertionError。
    """
    mgr = _new_manager(tmp_path)
    await mgr.close_session()

    with pytest.raises(AssertionError, match="no active session"):
        mgr.append_system_note("stray")
