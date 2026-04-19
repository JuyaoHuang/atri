"""Tests for src/service_context.py -- multi-character ChatAgent lifecycle.

Covers Phase 4 PRD US-AGT-005 acceptance criteria. Uses monkeypatch to stub
three dependencies at the ``src.service_context`` module namespace:
  * :func:`load_persona` -> returns a deterministic :class:`Persona`
  * :func:`create_from_role` -> returns a MagicMock LLM
  * :func:`_safe_build_long_term` -> returns a configurable LongTermMemory mock or None

The real :class:`src.memory.manager.MemoryManager` is constructed inside
:meth:`ServiceContext.get_or_create_agent` so its ``__init__`` path (bootstrap
session + short_term skeleton + chat_history metadata write) is exercised --
PRD says to stub only the persona / llm / long_term builders.

Note on PRD delta (batch discovered during US-AGT-005 implementation):
the PRD described ``LLMFactory.create_from_role`` (methodic form) but the
actual API is the module-level function :func:`src.llm.factory.create_from_role`.
ServiceContext imports it as ``create_from_role`` and tests patch that
same name in ``src.service_context`` -- PRD got the call shape right
logically but the attribute path wrong; the tests follow the code.

针对 src/service_context.py 的测试——多 character ChatAgent 生命周期。

覆盖 Phase 4 PRD US-AGT-005 的验收标准。通过 monkeypatch 在
``src.service_context`` 模块命名空间内替换三个依赖：
  * :func:`load_persona` -> 返回确定性的 :class:`Persona`
  * :func:`create_from_role` -> 返回 MagicMock LLM
  * :func:`_safe_build_long_term` -> 返回可配置的 LongTermMemory mock 或 None

真实的 :class:`src.memory.manager.MemoryManager` 会在
:meth:`ServiceContext.get_or_create_agent` 内部构造，因此其 ``__init__``
路径（自举会话 + short_term skeleton + chat_history metadata 写入）被实际
执行——PRD 要求仅 stub persona / llm / long_term 构建器。

PRD 偏差说明（US-AGT-005 实施期间发现）：PRD 写的是
``LLMFactory.create_from_role``（方法形式），但实际 API 是模块级函数
:func:`src.llm.factory.create_from_role`。ServiceContext 以
``create_from_role`` 名字导入，测试 patch ``src.service_context``
命名空间中的同一名字——PRD 的调用形态逻辑正确但属性路径写错；测试以代码为准。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.chat_agent import ChatAgent
from src.agent.persona import Persona
from src.service_context import ServiceContext

# ---------------------------------------------------------------------------
# Fixture helpers
# 固件辅助函数
# ---------------------------------------------------------------------------


def _persona_for(character_id: str) -> Persona:
    """Deterministic Persona factory so stubbed load_persona is predictable.

    确定性的 Persona 工厂，使 stub 的 load_persona 行为可预测。
    """
    return Persona(
        character_id=character_id,
        name=f"display_{character_id}",
        avatar=None,
        greeting=None,
        system_prompt=f"prompt-for-{character_id}",
    )


def _make_config(characters_dir: Path) -> dict[str, Any]:
    """Build a minimal but realistic config shape accepted by MemoryManager.

    ``storage.characters_dir`` points at the test tmp_path so no writes
    land in the real ``./data/characters/`` tree. The llm subsection is
    populated so ``create_from_role('chat', ...)`` would work if it were
    not stubbed.

    构造 MemoryManager 能接受的最小但真实的配置结构。

    ``storage.characters_dir`` 指向测试的 tmp_path，避免写入真实的
    ``./data/characters/`` 目录。llm 子节填得完整，这样即便不 stub，
    ``create_from_role('chat', ...)`` 也能跑通。
    """
    return {
        "llm": {
            "llm_configs": {
                "main_pool": {
                    "provider": "openai_compatible",
                    "model": "test-model",
                    "base_url": "http://unused.local",
                    "api_key": "unused",
                }
            },
            "llm_roles": {
                "chat": "main_pool",
                "l3_compress": "main_pool",
                "l4_compact": "main_pool",
            },
        },
        "memory": {
            "storage": {"characters_dir": str(characters_dir)},
            "short_term": {
                "snip": {"filler_words": []},
                "collapse": {
                    "trigger_rounds": 26,
                    "compress_rounds": 20,
                    "keep_recent_rounds": 6,
                },
                "super_compact": {"trigger_blocks": 4},
                "compressor": {"l3_role": "l3_compress", "l4_role": "l4_compact"},
            },
            "mem0": {"mode": "sdk", "sdk": {"api_key": "fake"}},
        },
    }


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    long_term: MagicMock | None = None,
) -> MagicMock:
    """Replace load_persona / create_from_role / _safe_build_long_term on
    ``src.service_context`` with test doubles.

    The patched names are the ones **service_context actually imports**
    (module-level), not whatever PRD described. Returns the shared stub LLM
    so tests can make assertions on it if needed.

    将 ``src.service_context`` 中的 load_persona / create_from_role /
    _safe_build_long_term 替换为测试替身。

    被 patch 的名字是 service_context **实际 import** 的那些（模块级），而不是
    PRD 描述的形式。返回共享的 stub LLM，供测试按需断言。
    """
    stub_llm = MagicMock(name="stub_llm")

    def _fake_load_persona(character_id: str) -> Persona:
        return _persona_for(character_id)

    def _fake_create_from_role(role: str, llm_config: dict[str, Any]) -> MagicMock:
        return stub_llm

    def _fake_safe_build_long_term(mem0_config: dict[str, Any]) -> MagicMock | None:
        return long_term

    monkeypatch.setattr("src.service_context.load_persona", _fake_load_persona)
    monkeypatch.setattr("src.service_context.create_from_role", _fake_create_from_role)
    monkeypatch.setattr("src.service_context._safe_build_long_term", _fake_safe_build_long_term)
    return stub_llm


def _make_long_term_mock() -> MagicMock:
    """Minimal LongTermMemory stub -- only ``close`` is exercised in tests.

    最小的 LongTermMemory stub——测试中只触及 ``close``。
    """
    lt = MagicMock(name="long_term")
    lt.close = MagicMock()
    return lt


# ---------------------------------------------------------------------------
# Lazy-load + cache (get_or_create_agent)
# 懒加载 + 缓存（get_or_create_agent）
# ---------------------------------------------------------------------------


def test_same_key_returns_same_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls with the same (character_id, user_id) tuple yield the same
    ChatAgent instance (cache hit; identity, not structural equality).

    相同 (character_id, user_id) 元组的两次调用返回同一个 ChatAgent 实例
    （缓存命中；identity 相等，非结构相等）。
    """
    _install_stubs(monkeypatch)
    ctx = ServiceContext(_make_config(tmp_path))

    first = ctx.get_or_create_agent("atri", "alice")
    second = ctx.get_or_create_agent("atri", "alice")

    assert first is second
    assert isinstance(first, ChatAgent)


def test_different_character_id_yields_distinct_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different character_id -> different ChatAgent instances; each carries
    its own Persona.

    不同 character_id 得到独立 ChatAgent 实例，各自持有对应 Persona。
    """
    _install_stubs(monkeypatch)
    ctx = ServiceContext(_make_config(tmp_path))

    atri = ctx.get_or_create_agent("atri", "alice")
    shizuku = ctx.get_or_create_agent("shizuku", "alice")

    assert atri is not shizuku
    assert atri.persona.character_id == "atri"
    assert shizuku.persona.character_id == "shizuku"


def test_same_character_different_user_yields_distinct_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache key is the **tuple** (character_id, user_id) -- same character
    with different user_id must give two MemoryManager instances.

    缓存键是 **元组** (character_id, user_id)——同 character 不同 user_id
    必须返回两个不同的 MemoryManager 实例。
    """
    _install_stubs(monkeypatch)
    ctx = ServiceContext(_make_config(tmp_path))

    alice_agent = ctx.get_or_create_agent("atri", "alice")
    bob_agent = ctx.get_or_create_agent("atri", "bob")

    assert alice_agent is not bob_agent
    assert alice_agent.memory_manager.user_id == "alice"
    assert bob_agent.memory_manager.user_id == "bob"


def test_memory_manager_wired_with_expected_character_and_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MemoryManager receives character_id and user_id from the call args.

    MemoryManager 接收的 character 和 user_id 来自调用参数。
    """
    _install_stubs(monkeypatch)
    ctx = ServiceContext(_make_config(tmp_path))

    agent = ctx.get_or_create_agent("atri", "alice")

    assert agent.memory_manager.character == "atri"
    assert agent.memory_manager.user_id == "alice"


def test_agent_builds_when_safe_build_long_term_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """long_term=None is a first-class supported state -- agent still builds.

    long_term=None 是一等支持的状态——agent 仍可构造。
    """
    _install_stubs(monkeypatch, long_term=None)
    ctx = ServiceContext(_make_config(tmp_path))

    agent = ctx.get_or_create_agent("atri", "alice")

    assert agent.memory_manager.long_term is None


# ---------------------------------------------------------------------------
# close_all -- graceful shutdown
# close_all——优雅关闭
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_all_awaits_close_session_on_every_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """close_all awaits close_session once per cached agent.

    close_all 对每个缓存的 agent 各 await 一次 close_session。
    """
    long_term = _make_long_term_mock()
    _install_stubs(monkeypatch, long_term=long_term)
    ctx = ServiceContext(_make_config(tmp_path))

    agents = [
        ctx.get_or_create_agent("atri", "alice"),
        ctx.get_or_create_agent("atri", "bob"),
        ctx.get_or_create_agent("shizuku", "alice"),
    ]
    for agent in agents:
        agent.memory_manager.close_session = AsyncMock()

    await ctx.close_all()

    for agent in agents:
        agent.memory_manager.close_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_all_calls_long_term_close_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """long_term.close() is called once per agent whose long_term is set.

    每个持有 long_term 的 agent 触发一次 long_term.close()。
    """
    long_term = _make_long_term_mock()
    _install_stubs(monkeypatch, long_term=long_term)
    ctx = ServiceContext(_make_config(tmp_path))

    for cid in ("atri", "shizuku"):
        agent = ctx.get_or_create_agent(cid, "alice")
        agent.memory_manager.close_session = AsyncMock()

    await ctx.close_all()

    assert long_term.close.call_count == 2


@pytest.mark.asyncio
async def test_close_all_tolerates_close_session_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising close_session must not stop the rest of the agents from
    being closed (best-effort shutdown semantic).

    单个 close_session 抛错不得阻断其余 agent 的关闭（best-effort 语义）。
    """
    long_term = _make_long_term_mock()
    _install_stubs(monkeypatch, long_term=long_term)
    ctx = ServiceContext(_make_config(tmp_path))

    a1 = ctx.get_or_create_agent("atri", "alice")
    a2 = ctx.get_or_create_agent("atri", "bob")
    a1.memory_manager.close_session = AsyncMock(side_effect=RuntimeError("boom"))
    a2.memory_manager.close_session = AsyncMock()

    await ctx.close_all()

    a1.memory_manager.close_session.assert_awaited_once()
    a2.memory_manager.close_session.assert_awaited_once()
    # long_term.close was still attempted for both agents (per-agent try/except)
    # 两个 agent 都尝试了 long_term.close（每个 agent 独立的 try/except）
    assert long_term.close.call_count == 2


@pytest.mark.asyncio
async def test_close_all_tolerates_long_term_close_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising long_term.close must not abort close_all iteration.

    long_term.close 抛错不得中断 close_all 循环。
    """
    long_term = _make_long_term_mock()
    long_term.close = MagicMock(side_effect=RuntimeError("close failed"))
    _install_stubs(monkeypatch, long_term=long_term)
    ctx = ServiceContext(_make_config(tmp_path))

    a1 = ctx.get_or_create_agent("atri", "alice")
    a2 = ctx.get_or_create_agent("atri", "bob")
    a1.memory_manager.close_session = AsyncMock()
    a2.memory_manager.close_session = AsyncMock()

    # 即便 long_term.close 两次都抛错，close_all 也必须完成
    # close_all must complete even though long_term.close raises twice.
    await ctx.close_all()

    a1.memory_manager.close_session.assert_awaited_once()
    a2.memory_manager.close_session.assert_awaited_once()
    assert long_term.close.call_count == 2


@pytest.mark.asyncio
async def test_close_all_skips_long_term_close_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """long_term=None branch: only close_session is awaited, no hasattr/close
    call is made on None.

    long_term=None 分支：只 await close_session，不对 None 调用 close。
    """
    _install_stubs(monkeypatch, long_term=None)
    ctx = ServiceContext(_make_config(tmp_path))

    agent = ctx.get_or_create_agent("atri", "alice")
    agent.memory_manager.close_session = AsyncMock()

    await ctx.close_all()

    agent.memory_manager.close_session.assert_awaited_once()
    assert agent.memory_manager.long_term is None


@pytest.mark.asyncio
async def test_close_all_on_empty_cache_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """close_all on a ServiceContext with no cached agents is a safe no-op.

    缓存为空时 close_all 是安全的 no-op。
    """
    _install_stubs(monkeypatch)
    ctx = ServiceContext(_make_config(tmp_path))

    # 不应抛任何异常
    # Must not raise.
    await ctx.close_all()
