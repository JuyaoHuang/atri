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
"""

from __future__ import annotations

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
    """Return ``{YYYY-MM-DD}_{8-hex}`` per §5.4 session-id convention."""
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"{date}_{secrets.token_hex(4)}"


def _is_valid_round(ai_msg: dict[str, Any]) -> bool:
    """Return True iff the AI reply counts toward ``total_rounds`` per §3.2.

    Rules (§3.2 count_rounds):
      * ``role == 'ai'``
      * content is truthy
      * content does not start with ``Error``
    """
    if ai_msg.get("role") != "ai":
        return False
    content = ai_msg.get("content") or ""
    if not content:
        return False
    return not content.startswith("Error")


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

        self._bootstrap_default_session()

    # ------------------------------------------------------------------
    # Session bootstrap (explicit start_session lives in US-MEM-007)
    # ------------------------------------------------------------------

    def _bootstrap_default_session(self) -> None:
        """Create an implicit session so ``on_round_complete`` works now."""
        self._init_session(_new_session_id())

    def _init_session(self, session_id: str) -> None:
        self._active_session_id = session_id
        self.short_term_store = ShortTermStore(self.character_dir, session_id, self.character)
        self.chat_history = ChatHistoryWriter(self.character_dir, session_id, self.character)
        state = self.short_term_store.load()
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
        self.chat_history.ensure_metadata()

    @property
    def state(self) -> dict[str, Any]:
        """Access the live short-term state dict (tests + read-only callers)."""
        assert self._state is not None, "MemoryManager has no active session"
        return self._state

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    # ------------------------------------------------------------------
    # Round processing
    # ------------------------------------------------------------------

    async def on_round_complete(
        self,
        user_msg: dict[str, Any],
        ai_msg: dict[str, Any],
    ) -> None:
        """Ingest one user/AI exchange.

        Applies L1 snip, appends to chat_history, updates short-term state
        (valid rounds only), evaluates L3/L4 triggers and persists.
        """
        assert self._state is not None
        assert self.chat_history is not None
        assert self.short_term_store is not None

        # L1 snip applies to user_msg only (AI replies stay untouched).
        cleaned_list, _freed = snip([user_msg], self.snip_config)
        cleaned_user = cleaned_list[0] if cleaned_list else user_msg

        # Append both turns to chat_history. Errors are kept for frontend
        # visibility even though they don't count as rounds.
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

        total_rounds: int = int(self._state["total_rounds"])
        if total_rounds > 0 and total_rounds % self.trigger_rounds == 0:
            await self._trigger_l3()

        if len(self._state["active_blocks"]) >= self.trigger_blocks:
            await self._trigger_l4()

        self.short_term_store.save(self._state)

    # ------------------------------------------------------------------
    # L3 / L4 triggers
    # ------------------------------------------------------------------

    def _next_uncompressed_round(self) -> int:
        """First round number not yet absorbed into any block."""
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
        self._state["meta_blocks"].insert(0, meta_block)
        self._state["active_blocks"] = active[self.trigger_blocks :]

        logger.info(
            f"L4 fired | meta_block_id={meta_block['block_id']} | "
            f"meta_blocks={len(self._state['meta_blocks'])} | "
            f"active_blocks={len(self._state['active_blocks'])}"
        )

    # ------------------------------------------------------------------
    # Long-term memory query
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
        """
        if self.long_term is None:
            return []
        return await self.long_term.search(
            query,
            user_id=self.user_id,
            agent_id=self.character,
            limit=limit,
        )


__all__ = ["MemoryManager", "LLMFactoryFn"]
