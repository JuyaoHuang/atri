"""Memory Manager -- per-round orchestration of short-term memory.

Responsibilities (Phase 3 Step 5 + Step 6 scope, per US-MEM-005/006):

* Apply L1 snip to each inbound user message.
* Append every exchange to the session's ``chat_history`` (both valid and
  error rounds so the frontend can render them).
* Track ``total_rounds`` and maintain ``recent_messages`` -- valid rounds
  only, per §3.2 round definition.
* Trigger L3 Collapse every ``trigger_rounds`` (compress the oldest
  ``compress_rounds`` rounds in ``recent_messages``). When a
  :class:`LongTermMemory` is injected, the same raw window is persisted to
  mem0 via ``long_term.add`` (best-effort; errors log WARNING but never
  break the round).
* Trigger L4 Super-Compact when ``len(active_blocks) >= trigger_blocks``.
* Expose :meth:`search_long_term` so callers can retrieve related facts
  before composing the LLM context.
* Persist short-term state via :class:`ShortTermStore` once per round.

Session lifecycle (``start_session`` / ``close_session``) and resume are
US-MEM-007/US-MEM-008.

Reference: docs/记忆系统设计讨论.md §3.2, §3.3, §4.2, §6.1.

Memory Manager——按轮次编排短期记忆。

职责范围（Phase 3 Step 5 + Step 6，对应 US-MEM-005/006）：

* 对每条进入的用户消息执行 L1 snip。
* 将每次交换追加到会话的 ``chat_history``（有效轮次与错误轮次都记录，
  以便前端渲染）。
* 跟踪 ``total_rounds`` 并维护 ``recent_messages``——仅统计有效轮次，
  轮次定义参见 §3.2。
* 每 ``trigger_rounds`` 触发一次 L3 Collapse（压缩 ``recent_messages``
  中最早的 ``compress_rounds`` 轮）。当注入了 :class:`LongTermMemory`
  时，同一原始窗口会通过 ``long_term.add`` 持久化到 mem0（尽力而为；
  失败仅记录 WARNING，绝不中断当前轮次）。
* 当 ``len(active_blocks) >= trigger_blocks`` 时触发 L4 Super-Compact。
* 暴露 :meth:`search_long_term` 供调用方在组装 LLM 上下文前检索相关事实。
* 每轮通过 :class:`ShortTermStore` 持久化短期状态一次。

会话生命周期（``start_session`` / ``close_session``）和恢复分别对应
US-MEM-007/US-MEM-008。

参考：docs/记忆系统设计讨论.md §3.2、§3.3、§4.2、§6.1。
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.llm.interface import LLMInterface
from src.memory.chat_history import ChatHistoryWriter
from src.memory.compressor import l3_collapse, l4_super_compact
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermStore
from src.memory.snip import snip

LLMFactoryFn = Callable[[str], LLMInterface]


def _new_session_id() -> str:
    """Return ``{YYYY-MM-DD}_{8-hex}`` per §5.4 session-id convention.

    按 §5.4 的会话 ID 规范返回 ``{YYYY-MM-DD}_{8-hex}``。
    """
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"{date}_{secrets.token_hex(4)}"


def _is_valid_round(ai_msg: dict[str, Any]) -> bool:
    """Return True iff the AI reply counts toward ``total_rounds`` per §3.2.

    Rules (§3.2 count_rounds):
      * ``role == 'ai'``
      * content is truthy
      * content does not start with ``Error``

    仅当 AI 回复按 §3.2 计入 ``total_rounds`` 时才返回 True。

    规则（§3.2 count_rounds）：
      * ``role == 'ai'``
      * content 为真值
      * content 不以 ``Error`` 开头
    """
    if ai_msg.get("role") != "ai":
        return False
    content = ai_msg.get("content") or ""
    if not content:
        return False
    return not content.startswith("Error")


_ROLE_MAP: dict[str, str] = {"human": "user", "ai": "assistant", "system": "system"}


def _map_role(role: str) -> str:
    """Translate internal role labels to the OpenAI-style API vocabulary.

    ``human`` -> ``user``, ``ai`` -> ``assistant``, ``system`` -> ``system``.
    Any other label raises :class:`ValueError` so accidental typos surface
    loudly instead of silently corrupting the LLM payload.

    将内部角色标签转换为 OpenAI 风格的 API 词汇。

    ``human`` -> ``user``、``ai`` -> ``assistant``、``system`` -> ``system``。
    任何其他标签会抛出 :class:`ValueError`，以便意外的拼写错误能显式暴露，
    而不是静默地损坏 LLM 载荷。
    """
    mapped = _ROLE_MAP.get(role)
    if mapped is None:
        raise ValueError(f"Unknown role: {role!r}")
    return mapped


class MemoryManager:
    """Central orchestrator of short-term memory for one character session.

    Responsibilities for Phase 3 Step 5:
      1. L1 snip the user message.
      2. Append both turns to chat_history (regardless of round validity).
      3. Update ``recent_messages`` + ``total_rounds`` (valid rounds only).
      4. Trigger L3 Collapse when ``total_rounds % trigger_rounds == 0``.
      5. Trigger L4 Super-Compact when ``len(active_blocks) >= trigger_blocks``.
      6. Persist via :class:`ShortTermStore` once per round.

    The constructor bootstraps a default session so tests and demos can call
    :meth:`on_round_complete` immediately. US-MEM-007's ``start_session``
    will introduce the explicit entry point.

    单个角色会话的短期记忆中央编排器。

    Phase 3 Step 5 的职责：
      1. 对用户消息执行 L1 snip。
      2. 将双方消息追加到 chat_history（无论轮次是否有效）。
      3. 更新 ``recent_messages`` + ``total_rounds``（仅统计有效轮次）。
      4. 当 ``total_rounds % trigger_rounds == 0`` 时触发 L3 Collapse。
      5. 当 ``len(active_blocks) >= trigger_blocks`` 时触发 L4 Super-Compact。
      6. 每轮通过 :class:`ShortTermStore` 持久化一次。

    构造器会自举一个默认会话，使测试和示例能够立即调用
    :meth:`on_round_complete`。US-MEM-007 的 ``start_session`` 将提供显式
    的入口点。
    """

    def __init__(
        self,
        memory_config: dict[str, Any],
        llm_factory_fn: LLMFactoryFn,
        character: str,
        user_id: str,
        character_dir: Path | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self.memory_config = memory_config
        self.llm_factory_fn = llm_factory_fn
        self.character = character
        self.user_id = user_id
        self.long_term = long_term

        short_term_cfg = memory_config.get("short_term", {})
        collapse_cfg = short_term_cfg.get("collapse", {})
        super_cfg = short_term_cfg.get("super_compact", {})
        compressor_cfg = short_term_cfg.get("compressor", {})

        self.trigger_rounds: int = int(collapse_cfg.get("trigger_rounds", 26))
        self.compress_rounds: int = int(collapse_cfg.get("compress_rounds", 20))
        self.keep_recent_rounds: int = int(collapse_cfg.get("keep_recent_rounds", 6))
        self.trigger_blocks: int = int(super_cfg.get("trigger_blocks", 4))
        self.snip_config: dict[str, Any] = dict(short_term_cfg.get("snip", {}))
        self.l3_role: str = compressor_cfg.get("l3_role", "l3_compress")
        self.l4_role: str = compressor_cfg.get("l4_role", "l4_compact")

        if character_dir is None:
            chars_root = Path(
                memory_config.get("storage", {}).get("characters_dir", "./data/characters")
            )
            character_dir = chars_root / character
        self.character_dir = Path(character_dir)

        self._active_session_id: str | None = None
        self._state: dict[str, Any] | None = None
        self.short_term_store: ShortTermStore | None = None
        self.chat_history: ChatHistoryWriter | None = None
        # ``_dirty`` = "there are rounds in recent_messages added during this
        # session's lifetime that haven't been pushed to long-term yet". L3
        # pushes only the compressed window, so the retained tail still counts
        # as dirty until close_session flushes it (or a follow-up L3 does).
        # ``_dirty`` = “recent_messages 中在本次会话生命周期内新增、但尚未推送
        # 到长期记忆的轮次”。L3 只推送被压缩的窗口，所以保留的尾部仍计为 dirty，
        # 直到 close_session 将其刷出（或后续 L3 带走）。
        self._dirty: bool = False

        self._bootstrap_default_session()

    # ------------------------------------------------------------------
    # Session bootstrap (explicit start_session lives in US-MEM-007)
    # 会话自举（显式的 start_session 在 US-MEM-007 中实现）
    # ------------------------------------------------------------------

    def _bootstrap_default_session(self) -> None:
        """Create an implicit session so ``on_round_complete`` works now.

        创建一个隐式会话，使 ``on_round_complete`` 能够立即工作。
        """
        self._init_session(_new_session_id())

    def _init_session(self, session_id: str) -> None:
        self._active_session_id = session_id
        self.short_term_store = ShortTermStore(self.character_dir, session_id, self.character)
        self.chat_history = ChatHistoryWriter(self.character_dir, session_id, self.character)
        try:
            state = self.short_term_store.load()
        except (json.JSONDecodeError, ValueError) as exc:
            # Bootstrap/start must never crash on a pre-existing corrupt file.
            # Explicit recovery belongs to :meth:`resume_session`, which runs a
            # full chat_history replay when it detects corruption itself.
            # 自举/启动过程绝不能因为已存在的损坏文件而崩溃。显式恢复属于
            # :meth:`resume_session` 的职责，它在检测到损坏时会执行完整的
            # chat_history 回放。
            logger.warning(
                f"short_term unparseable at session init | session_id={session_id} | "
                f"{exc!r}; using fresh skeleton"
            )
            state = ShortTermStore.get_skeleton(session_id, self.character)
        state["session_id"] = session_id
        state["character"] = self.character
        state.setdefault("total_rounds", 0)
        state.setdefault("meta_blocks", [])
        state.setdefault("active_blocks", [])
        state.setdefault("recent_messages", [])
        self._state = state
        # Write the metadata row eagerly so the chat_history file is always
        # well-formed, even if the first round never lands (crash between
        # start and first turn).
        # 提前写入 metadata 行，使 chat_history 文件始终保持良好结构，即便第一
        # 轮从未到达（启动与首轮之间发生崩溃）也不至于格式异常。
        self.chat_history.ensure_metadata()

    @property
    def state(self) -> dict[str, Any]:
        """Access the live short-term state dict (tests + read-only callers).

        访问实时的短期状态字典（供测试和只读调用方使用）。
        """
        assert self._state is not None, "MemoryManager has no active session"
        return self._state

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    # ------------------------------------------------------------------
    # Explicit session lifecycle (US-MEM-007)
    # 显式会话生命周期（US-MEM-007）
    # ------------------------------------------------------------------

    async def start_session(self) -> str:
        """Open a fresh session, overriding any bootstrap one.

        Generates a new ``session_id`` (``{YYYY-MM-DD}_{8-hex}``), rebuilds
        the :class:`ShortTermStore` / :class:`ChatHistoryWriter` bindings,
        writes the chat_history metadata row, and resets the dirty flag so
        the next ``close_session`` only flushes rounds added during this
        lifetime.

        打开一个全新会话，覆盖任何自举生成的会话。

        生成新的 ``session_id``（``{YYYY-MM-DD}_{8-hex}``），重建
        :class:`ShortTermStore` / :class:`ChatHistoryWriter` 绑定，写入
        chat_history 的 metadata 行，并重置 dirty 标志，使下一次
        ``close_session`` 只刷出本生命周期内新增的轮次。
        """
        session_id = _new_session_id()
        self._init_session(session_id)
        self._dirty = False
        logger.info(
            f"MemoryManager session started | session_id={session_id} | "
            f"character={self.character} | user_id={self.user_id}"
        )
        return session_id

    async def close_session(self) -> None:
        """Flush remaining uncompressed messages and release session state.

        Writes any ``recent_messages`` added since the last L3 / long-term
        push to mem0 (best-effort) and persists the short-term state one
        last time. When no new rounds arrived this lifetime the long-term
        write is skipped entirely to avoid empty / duplicate writes
        (§4.4 idempotency). After return the manager has no active session;
        callers must invoke :meth:`start_session` or
        :meth:`resume_session` before the next round.

        刷出剩余的未压缩消息并释放会话状态。

        将自上次 L3 / 长期推送以来新增的 ``recent_messages`` 写入 mem0
        （尽力而为），并对短期状态进行最后一次持久化。若本生命周期内没有
        新增轮次，则完全跳过长期写入，避免空写/重复写（§4.4 幂等性）。
        返回后管理器没有活动会话；调用方必须在下一轮之前调用
        :meth:`start_session` 或 :meth:`resume_session`。
        """
        if self._state is None:
            return

        session_id = self._active_session_id
        recent: list[dict[str, Any]] = list(self._state.get("recent_messages", []))
        pushed_tail = False
        if self._dirty and recent and self.long_term is not None:
            await self.long_term.add(
                recent,
                user_id=self.user_id,
                agent_id=self.character,
                run_id=session_id or "default",
            )
            pushed_tail = True

        if self.short_term_store is not None:
            self.short_term_store.save(self._state)

        logger.info(
            f"MemoryManager session closed | session_id={session_id} | pushed_tail={pushed_tail}"
        )

        self._dirty = False
        self._active_session_id = None
        self._state = None
        self.short_term_store = None
        self.chat_history = None

    # ------------------------------------------------------------------
    # Session resume (US-MEM-008)
    # 会话恢复（US-MEM-008）
    # ------------------------------------------------------------------

    async def resume_session(self, session_id: str) -> None:
        """Reopen a prior session, reconciling stored state with chat_history.

        Strategy (§8.5):

        1. Rebind ``ShortTermStore`` / ``ChatHistoryWriter`` to ``session_id``.
        2. Try to load ``short_term_memory.json``. On JSON corruption fall
           through to a full rebuild.
        3. Count valid rounds from ``chat_history``. Compare against the
           stored ``total_rounds``.
        4. Consistent -> restore as-is.
        5. Behind by N -> append the missing tail to ``recent_messages`` and
           bump ``total_rounds``. If the catch-up lands on a trigger
           boundary, fire L3 (chaining L4 when 4+ blocks accumulate).
        6. Stored ahead of chat_history -> log WARNING (unexpected) and
           keep stored state.

        The mem0 ADD-only semantics (v3) make the L3 re-push during rebuild
        idempotent; duplicate facts cost LLM tokens but don't corrupt state.

        重新打开先前的会话，对齐存储状态与 chat_history。

        策略（§8.5）：

        1. 将 ``ShortTermStore`` / ``ChatHistoryWriter`` 重新绑定到
           ``session_id``。
        2. 尝试加载 ``short_term_memory.json``。遇到 JSON 损坏则退化为
           完全重建。
        3. 从 ``chat_history`` 统计有效轮次，与存储的 ``total_rounds`` 对比。
        4. 一致 -> 原样恢复。
        5. 落后 N 轮 -> 将缺失的尾部追加到 ``recent_messages`` 并增加
           ``total_rounds``。若追补落在触发边界上，则触发 L3
           （当累计 4+ 个块时级联触发 L4）。
        6. 存储超前于 chat_history -> 记录 WARNING（不应发生），保留存储状态。

        mem0 v3 的纯 ADD 语义使得重建期间重复推送 L3 是幂等的；重复事实会
        耗费一些 LLM token，但不会污染状态。
        """
        # Tear down anything the constructor's bootstrap left active.
        # 拆除构造器自举留下的活动状态。
        self._dirty = False
        self._state = None
        self.short_term_store = None
        self.chat_history = None

        self._active_session_id = session_id
        self.short_term_store = ShortTermStore(self.character_dir, session_id, self.character)
        self.chat_history = ChatHistoryWriter(self.character_dir, session_id, self.character)

        try:
            state = self.short_term_store.load()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                f"short_term unparseable for session_id={session_id}: {exc!r}; "
                "running full rebuild from chat_history"
            )
            await self._full_rebuild_from_chat_history()
            return

        state.setdefault("session_id", session_id)
        state.setdefault("character", self.character)
        state.setdefault("total_rounds", 0)
        state.setdefault("meta_blocks", [])
        state.setdefault("active_blocks", [])
        state.setdefault("recent_messages", [])
        self._state = state

        pairs = self._collect_chat_history_pairs()
        chat_rounds = len(pairs)
        stored_rounds = int(state.get("total_rounds", 0))

        if chat_rounds == stored_rounds:
            logger.info(
                f"Session resumed consistent | session_id={session_id} | rounds={stored_rounds}"
            )
            return

        if chat_rounds < stored_rounds:
            logger.warning(
                f"Stored total_rounds ({stored_rounds}) exceeds chat_history "
                f"({chat_rounds}) for session_id={session_id}; keeping stored state"
            )
            return

        # Incremental tail replay.
        # 增量回放尾部缺失的轮次。
        missing = pairs[stored_rounds:]
        for human_msg, ai_msg in missing:
            cleaned_list, _ = snip(
                [{"role": "human", "content": human_msg.get("content", "")}],
                self.snip_config,
            )
            cleaned = cleaned_list[0] if cleaned_list else human_msg
            state["recent_messages"].append(
                {"role": "human", "content": cleaned.get("content", "")}
            )
            state["recent_messages"].append({"role": "ai", "content": ai_msg.get("content", "")})
            state["total_rounds"] = int(state.get("total_rounds", 0)) + 1

            total = int(state["total_rounds"])
            if total > 0 and total % self.trigger_rounds == 0:
                await self._trigger_l3()
            if len(state["active_blocks"]) >= self.trigger_blocks:
                await self._trigger_l4()

        self.short_term_store.save(state)
        # Catch-up only replays already-persisted chat_history; no new data
        # to flush at close_session, so clear the dirty flag.
        # 追补只是回放已持久化的 chat_history；close_session 时没有新数据需要
        # 刷出，因此清除 dirty 标志。
        self._dirty = False
        logger.info(
            f"Session resumed with catch-up | session_id={session_id} | "
            f"stored={stored_rounds} chat={chat_rounds}"
        )

    async def _full_rebuild_from_chat_history(self) -> None:
        """Rebuild short-term state from an empty skeleton.

        Replays every (human, ai) pair through the same L1 + recent_messages
        + trigger pipeline ``on_round_complete`` uses. L3 windows also hit
        ``long_term.add`` (mem0 v3 is ADD-only so duplicates are idempotent).

        从空骨架重建短期状态。

        将每一对 (human, ai) 经由与 ``on_round_complete`` 一致的 L1 +
        recent_messages + 触发流水线重放。L3 窗口也会调用 ``long_term.add``
        （mem0 v3 仅支持 ADD，因此重复写是幂等的）。
        """
        assert self._active_session_id is not None
        assert self.short_term_store is not None
        assert self.chat_history is not None

        state = ShortTermStore.get_skeleton(self._active_session_id, self.character)
        self._state = state

        for human_msg, ai_msg in self._collect_chat_history_pairs():
            cleaned_list, _ = snip(
                [{"role": "human", "content": human_msg.get("content", "")}],
                self.snip_config,
            )
            cleaned = cleaned_list[0] if cleaned_list else human_msg
            state["recent_messages"].append(
                {"role": "human", "content": cleaned.get("content", "")}
            )
            state["recent_messages"].append({"role": "ai", "content": ai_msg.get("content", "")})
            state["total_rounds"] = int(state.get("total_rounds", 0)) + 1

            total = int(state["total_rounds"])
            if total > 0 and total % self.trigger_rounds == 0:
                await self._trigger_l3()
            if len(state["active_blocks"]) >= self.trigger_blocks:
                await self._trigger_l4()

        self.short_term_store.save(state)
        self._dirty = False
        logger.info(
            f"Session fully rebuilt | session_id={self._active_session_id} | "
            f"rounds={state['total_rounds']}"
        )

    def _collect_chat_history_pairs(
        self,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Group chat_history into (human, ai) pairs where ai is a valid round.

        Metadata / system entries are skipped. Unpaired human messages (e.g.
        an incomplete trailing record) are dropped. Invalid AI replies
        (``Error``-prefixed content) are also excluded, matching §3.2.

        将 chat_history 分组为 (human, ai) 对，其中 ai 必须是有效轮次。

        跳过 metadata / system 条目。未配对的 human 消息（例如不完整的尾部
        记录）会被丢弃。无效的 AI 回复（内容以 ``Error`` 开头）也会被排除，
        与 §3.2 保持一致。
        """
        assert self.chat_history is not None
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        pending_human: dict[str, Any] | None = None
        for entry in self.chat_history.iter_messages():
            role = entry.get("role")
            if role == "human":
                pending_human = entry
            elif role == "ai":
                if pending_human is not None and _is_valid_round(entry):
                    pairs.append((pending_human, entry))
                pending_human = None
        return pairs

    async def on_round_complete(
        self,
        user_msg: dict[str, Any],
        ai_msg: dict[str, Any],
    ) -> None:
        """Ingest one user/AI exchange.

        Applies L1 snip, appends to chat_history, updates short-term state
        (valid rounds only), evaluates L3/L4 triggers and persists.

        接收一次 user/AI 交换。

        应用 L1 snip，追加到 chat_history，更新短期状态（仅统计有效轮次），
        评估 L3/L4 触发条件并持久化。
        """
        assert self._state is not None
        assert self.chat_history is not None
        assert self.short_term_store is not None

        # L1 snip applies to user_msg only (AI replies stay untouched).
        # L1 snip 仅作用于 user_msg（AI 回复保持不变）。
        cleaned_list, _freed = snip([user_msg], self.snip_config)
        cleaned_user = cleaned_list[0] if cleaned_list else user_msg

        # Append both turns to chat_history. Errors are kept for frontend
        # visibility even though they don't count as rounds.
        # 双方消息都追加到 chat_history。错误消息虽然不计入轮次，但仍保留以
        # 便前端可见。
        self.chat_history.append_human(
            cleaned_user.get("content", ""),
            raw_input=user_msg.get("raw_input"),
            name=cleaned_user.get("name", "user"),
        )
        self.chat_history.append_ai(
            ai_msg.get("content", ""),
            name=ai_msg.get("name", self.character),
            avatar=ai_msg.get("avatar"),
        )

        if _is_valid_round(ai_msg):
            self._state["recent_messages"].append(
                {"role": "human", "content": cleaned_user.get("content", "")}
            )
            self._state["recent_messages"].append(
                {"role": "ai", "content": ai_msg.get("content", "")}
            )
            self._state["total_rounds"] = int(self._state.get("total_rounds", 0)) + 1
            self._dirty = True

        total_rounds: int = int(self._state["total_rounds"])
        if total_rounds > 0 and total_rounds % self.trigger_rounds == 0:
            await self._trigger_l3()

        if len(self._state["active_blocks"]) >= self.trigger_blocks:
            await self._trigger_l4()

        self.short_term_store.save(self._state)

    # ------------------------------------------------------------------
    # System-level annotations (Phase 4 error path)
    # 系统级标注（Phase 4 错误路径）
    # ------------------------------------------------------------------

    def append_system_note(self, content: str) -> None:
        """Append a ``role=system`` row to ``chat_history`` without touching state.

        Used by :class:`src.agent.chat_agent.ChatAgent`'s error path: when the
        LLM call raises :class:`src.llm.exceptions.LLMError`, the agent records
        the failure here so the frontend can render it, but the round is **not**
        counted -- ``total_rounds`` and ``recent_messages`` stay untouched, no
        L3/L4 trigger fires, and nothing outside the append itself is persisted
        (short-term state is unchanged, so no ``short_term_store.save``).

        Synchronous by design: :meth:`ChatHistoryWriter.append_system` is a
        plain file append, and callers already hold the event-loop context from
        their surrounding async path.

        在 chat_history 追加一条 ``role=system`` 记录，不影响任何状态。

        供 :class:`src.agent.chat_agent.ChatAgent` 错误路径使用：当 LLM 调用
        抛出 :class:`src.llm.exceptions.LLMError` 时，Agent 在此处记录失败，
        使前端能够渲染；但该轮**不**计数——``total_rounds`` 和
        ``recent_messages`` 保持不变，不触发 L3/L4，也不产生除 append 本身
        之外的持久化（short-term 状态未变，因此不会调用
        ``short_term_store.save``）。

        设计为同步方法：:meth:`ChatHistoryWriter.append_system` 仅是普通的
        文件追加，调用方的异步上下文由外部持有。
        """
        assert self.chat_history is not None, "MemoryManager has no active session"
        self.chat_history.append_system(content)

    # ------------------------------------------------------------------
    # L3 / L4 triggers
    # L3 / L4 触发器
    # ------------------------------------------------------------------

    def _next_uncompressed_round(self) -> int:
        """First round number not yet absorbed into any block.

        尚未被任何块吸收的最小轮次号。
        """
        assert self._state is not None
        max_end = 0
        for block in self._state["meta_blocks"] + self._state["active_blocks"]:
            end = int(block["covers_rounds"][1])
            if end > max_end:
                max_end = end
        return max_end + 1

    async def _trigger_l3(self) -> None:
        assert self._state is not None
        recent: list[dict[str, Any]] = self._state["recent_messages"]
        window_msg_count = self.compress_rounds * 2
        if len(recent) < window_msg_count:
            logger.warning(
                f"L3 requested but recent_messages has only {len(recent)} msgs "
                f"(need {window_msg_count}); skipping"
            )
            return

        window = recent[:window_msg_count]
        start_round = self._next_uncompressed_round()
        end_round = start_round + self.compress_rounds - 1

        llm = self.llm_factory_fn(self.l3_role)
        block = await l3_collapse(window, llm, start_round, end_round)
        self._state["active_blocks"].append(block)
        del recent[:window_msg_count]

        if self.long_term is not None:
            await self.long_term.add(
                window,
                user_id=self.user_id,
                agent_id=self.character,
                run_id=self._active_session_id or "default",
            )

        logger.info(
            f"L3 fired | rounds={start_round}-{end_round} | "
            f"block_id={block['block_id']} | "
            f"active_blocks={len(self._state['active_blocks'])}"
        )

    async def _trigger_l4(self) -> None:
        assert self._state is not None
        active: list[dict[str, Any]] = self._state["active_blocks"]
        source_blocks = active[: self.trigger_blocks]

        llm = self.llm_factory_fn(self.l4_role)
        meta_block = await l4_super_compact(source_blocks, llm)

        # Newest meta_block at index 0 (reverse-chronological convention so
        # downstream context builders can render oldest-first via reversed()).
        # 最新的 meta_block 放在索引 0（逆序存储约定，这样下游上下文构建器
        # 可通过 reversed() 按最旧优先顺序渲染）。
        self._state["meta_blocks"].insert(0, meta_block)
        self._state["active_blocks"] = active[self.trigger_blocks :]

        logger.info(
            f"L4 fired | meta_block_id={meta_block['block_id']} | "
            f"meta_blocks={len(self._state['meta_blocks'])} | "
            f"active_blocks={len(self._state['active_blocks'])}"
        )

    # ------------------------------------------------------------------
    # Long-term memory query
    # 长期记忆查询
    # ------------------------------------------------------------------

    async def search_long_term(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve related long-term facts for the current user/agent pair.

        Returns ``[]`` when no ``LongTermMemory`` was injected so callers can
        build the LLM context unconditionally. Threshold is handled inside
        :class:`LongTermMemory` (defaults mirror §8.3).

        为当前 user/agent 对检索相关的长期事实。

        当未注入 ``LongTermMemory`` 时返回 ``[]``，这样调用方可以无条件地
        构建 LLM 上下文。阈值在 :class:`LongTermMemory` 内部处理（默认值
        对齐 §8.3）。
        """
        if self.long_term is None:
            return []
        return await self.long_term.search(
            query,
            user_id=self.user_id,
            agent_id=self.character,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # LLM context builder (US-MEM-009, §3.5 payload order)
    # LLM 上下文构建器（US-MEM-009，§3.5 载荷顺序）
    # ------------------------------------------------------------------

    async def build_llm_context(
        self,
        user_input: str,
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Assemble the messages payload for the next LLM turn.

        Order follows design doc §3.5 strictly:

          1. ``system_prompt``                                (skipped if empty)
          2. Long-term facts retrieved via ``long_term.search`` wrapped in a
             single system message ``"关于这位用户，你记得：\\n- ..."``   (skipped when empty)
          3. Each ``meta_block['summary']``                   oldest-first
          4. Each ``active_block['summary']``                 oldest-first
          5. ``recent_messages`` expanded one message each, with role
             mapping ``human`` -> ``user`` and ``ai`` -> ``assistant``.
          6. ``user_input`` as a final ``{'role':'user', ...}``.

        This method does **not** call the LLM -- it only composes the list.
        ``ChatAgent`` (Phase 4) consumes the result and feeds it to
        :class:`LLMInterface`.

        为下一次 LLM 回合组装消息载荷。

        顺序严格遵循设计文档 §3.5：

          1. ``system_prompt``                                （为空时跳过）
          2. 通过 ``long_term.search`` 检索得到的长期事实，包装成单条系统消息
             ``"关于这位用户，你记得：\\n- ..."``                （无命中时跳过）
          3. 每个 ``meta_block['summary']``                    最旧优先
          4. 每个 ``active_block['summary']``                  最旧优先
          5. ``recent_messages`` 逐条展开，角色映射为
             ``human`` -> ``user``、``ai`` -> ``assistant``。
          6. ``user_input`` 作为最后一条 ``{'role':'user', ...}``。

        本方法**不**调用 LLM——它只负责组装列表。``ChatAgent``（Phase 4）
        会消费返回结果并将其喂给 :class:`LLMInterface`。
        """
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        long_term_hits = await self.search_long_term(user_input)
        if long_term_hits:
            facts = [str(h.get("memory", "")).strip() for h in long_term_hits]
            facts = [f for f in facts if f]
            if facts:
                joined = "\n- ".join(facts)
                messages.append(
                    {
                        "role": "system",
                        "content": f"关于这位用户，你记得：\n- {joined}",
                    }
                )

        if self._state is not None:
            # meta_blocks are stored newest-first (§4.2 convention); render
            # oldest-first so the LLM reads history chronologically.
            # meta_blocks 按最新优先存储（§4.2 约定）；渲染时改为最旧优先，
            # 这样 LLM 可以按时间顺序阅读历史。
            for meta in reversed(self._state.get("meta_blocks", [])):
                summary = meta.get("summary", "")
                if summary:
                    messages.append({"role": "system", "content": summary})
            for active in self._state.get("active_blocks", []):
                summary = active.get("summary", "")
                if summary:
                    messages.append({"role": "system", "content": summary})
            for raw in self._state.get("recent_messages", []):
                messages.append(
                    {
                        "role": _map_role(raw.get("role", "")),
                        "content": raw.get("content", ""),
                    }
                )

        messages.append({"role": "user", "content": user_input})
        return messages


__all__ = ["MemoryManager", "LLMFactoryFn"]
