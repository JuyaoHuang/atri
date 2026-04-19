"""ServiceContext -- top-level owner of ChatAgent lifecycle (Phase 4, US-AGT-005).

One ServiceContext per process. It holds the merged config (from
:func:`src.utils.config_loader.load_config`) and lazily constructs one
:class:`src.agent.chat_agent.ChatAgent` per ``(character_id, user_id)``
pair, caching each instance so subsequent requests reuse the same memory
state. Phase 5 FastAPI WebSocket handlers will route by ``character_id``
from the frontend session and call :meth:`get_or_create_agent`.

Key invariants (per docs/Phase4_执行规格.md §US-AGT-005 and decision S3):

* Cache key is the **tuple** ``(character_id, user_id)`` -- same character
  accessed by two users yields two independent :class:`MemoryManager`
  instances (distinct ``user_id`` -> distinct mem0 filters).
* ``_safe_build_long_term`` is best-effort: if mem0 construction fails
  (missing API key, Qdrant unreachable, etc.) it logs a WARNING and returns
  ``None``; the resulting ChatAgent still works with short-term memory only.
* :meth:`close_all` tolerates per-agent failures so one broken session does
  not block others from flushing.

Reference: docs/Phase4_执行规格.md §US-AGT-005, docs/项目架构设计.md §2.5,
docs/LLM调用层设计讨论.md §2.3 (role-based LLMFactory).

ServiceContext——ChatAgent 生命周期的顶层拥有者（Phase 4，US-AGT-005）。

每个进程持有一个 ServiceContext。它保存合并后的配置（来自
:func:`src.utils.config_loader.load_config`），并按 ``(character_id, user_id)``
元组懒加载构造 :class:`src.agent.chat_agent.ChatAgent`，缓存实例，使后续请求
复用同一份记忆状态。Phase 5 的 FastAPI WebSocket 处理器将根据前端会话的
``character_id`` 路由并调用 :meth:`get_or_create_agent`。

关键不变式（对齐 docs/Phase4_执行规格.md §US-AGT-005 和决策 S3）：

* 缓存键是 ``(character_id, user_id)`` **元组**——同一角色被两个用户访问
  会得到两个独立的 :class:`MemoryManager` 实例（不同的 ``user_id`` 对应
  不同的 mem0 过滤器）。
* ``_safe_build_long_term`` 尽力而为：若 mem0 构造失败（缺 API key、
  Qdrant 不可达等），记录 WARNING 并返回 ``None``；生成的 ChatAgent 仍能
  基于短期记忆工作。
* :meth:`close_all` 容忍单个 Agent 的失败，不让一处故障阻塞其他会话的
  刷出。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agent.chat_agent import ChatAgent
from src.agent.persona import load_persona
from src.llm.factory import create_from_role
from src.llm.interface import LLMInterface
from src.memory.long_term import LongTermMemory
from src.memory.manager import MemoryManager


def _safe_build_long_term(mem0_config: dict[str, Any]) -> LongTermMemory | None:
    """Best-effort :class:`LongTermMemory` construction.

    Migrated from ``src/main.py`` in US-AGT-006 so both the main-demo
    startup and the ServiceContext path use the same helper. Rationale:
    in ``local_deploy`` mode, ``Memory.from_config`` may ping Qdrant /
    Ollama; returning ``None`` on failure lets the rest of the app
    (short-term memory, LLM call, chat_history) still function.

    Args:
        mem0_config: The ``memory.mem0`` subsection of the merged config.

    Returns:
        A :class:`LongTermMemory` instance on success, or ``None`` if
        construction raised any exception.

    尽力而为地构造 :class:`LongTermMemory`。

    US-AGT-006 从 ``src/main.py`` 迁移而来，使主 demo 启动与 ServiceContext
    路径共用同一个 helper。理由：``local_deploy`` 模式下，
    ``Memory.from_config`` 可能会访问 Qdrant / Ollama；构造失败时返回
    ``None``，让应用的其余部分（短期记忆、LLM 调用、chat_history）仍能正常
    运行。

    参数：
        mem0_config：合并配置中的 ``memory.mem0`` 子节。

    返回：
        成功时返回 :class:`LongTermMemory` 实例；构造过程中抛出任何异常时
        返回 ``None``。
    """
    if not mem0_config:
        logger.warning("mem0 config missing; skipping LongTermMemory construction")
        return None
    try:
        return LongTermMemory(mem0_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LongTermMemory construction failed (continuing without long-term "
            "memory) | mode={} error={!r}",
            mem0_config.get("mode"),
            exc,
        )
        return None


class ServiceContext:
    """Top-level container managing ChatAgent instances per character/user.

    持有每个 character/user 对应的 ChatAgent 实例的顶层容器。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._agents: dict[tuple[str, str], ChatAgent] = {}

    def get_or_create_agent(self, character_id: str, user_id: str) -> ChatAgent:
        """Return the cached agent for ``(character_id, user_id)`` or build a new one.

        Creation path (US-AGT-005 PRD):
          1. ``persona = load_persona(character_id)``
          2. ``long_term = _safe_build_long_term(config['memory']['mem0'])``
          3. Build ``llm_factory_fn`` closure that defers to
             :func:`create_from_role` with the full ``llm`` config.
          4. Construct ``MemoryManager(config['memory'], llm_factory_fn,
             character=character_id, user_id=user_id, long_term=long_term)``.
          5. Build the main-chat LLM via ``create_from_role('chat', ...)``.
          6. ``agent = ChatAgent(chat_llm, mgr, persona)``.

        返回 ``(character_id, user_id)`` 对应的缓存 Agent；未命中则新建。

        创建路径（US-AGT-005 PRD）：
          1. ``persona = load_persona(character_id)``
          2. ``long_term = _safe_build_long_term(config['memory']['mem0'])``
          3. 构造 ``llm_factory_fn`` 闭包，委托给携带完整 ``llm`` 配置的
             :func:`create_from_role`。
          4. 构造 ``MemoryManager(config['memory'], llm_factory_fn,
             character=character_id, user_id=user_id, long_term=long_term)``。
          5. 通过 ``create_from_role('chat', ...)`` 构造主聊天 LLM。
          6. ``agent = ChatAgent(chat_llm, mgr, persona)``。
        """
        key = (character_id, user_id)
        cached = self._agents.get(key)
        if cached is not None:
            return cached

        llm_config = self.config.get("llm", {})
        memory_config = self.config.get("memory", {})
        mem0_config = memory_config.get("mem0", {})

        persona = load_persona(character_id)
        long_term = _safe_build_long_term(mem0_config)

        def _llm_factory_fn(role: str) -> LLMInterface:
            return create_from_role(role, llm_config)

        mgr = MemoryManager(
            memory_config,
            _llm_factory_fn,
            character=character_id,
            user_id=user_id,
            long_term=long_term,
        )
        chat_llm = create_from_role("chat", llm_config)
        agent = ChatAgent(chat_llm, mgr, persona)

        self._agents[key] = agent
        logger.info(
            "ChatAgent created | character={} | user_id={} | long_term={}",
            character_id,
            user_id,
            "on" if long_term is not None else "off",
        )
        return agent

    async def close_all(self) -> None:
        """Flush every cached agent's session and release long-term handles.

        Best-effort per agent: a failure in ``close_session`` or
        ``long_term.close`` is logged as WARNING but does not prevent the
        remaining agents from being closed. Safe to call during process
        shutdown even if some sessions never received a round.

        冲刷每个缓存 Agent 的会话并释放长期记忆句柄。

        对每个 Agent 尽力而为：``close_session`` 或 ``long_term.close``
        失败会以 WARNING 形式记录，但不会阻止其余 Agent 被关闭。即便某些
        会话从未接收过轮次，也可在进程关闭期间安全调用。
        """
        logger.info("ServiceContext close_all | agent_count={}", len(self._agents))
        for key, agent in self._agents.items():
            try:
                await agent.memory_manager.close_session()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "close_session failed during close_all | key={} error={!r}",
                    key,
                    exc,
                )
            long_term = agent.memory_manager.long_term
            if long_term is not None:
                try:
                    long_term.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "long_term.close failed during close_all | key={} error={!r}",
                        key,
                        exc,
                    )
        logger.info("ServiceContext close_all complete")


__all__ = ["ServiceContext", "_safe_build_long_term"]
