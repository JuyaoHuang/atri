"""L3 Collapse + L4 Super-Compact compression functions.

Both functions consume raw conversation data and call a supplied
``LLMInterface`` to produce structured summaries. The L3 function compresses
a window of ~20 rounds into one *event-level* block; L4 merges several L3
blocks into one *pattern-level* meta-block.

Reference: docs/记忆系统设计讨论.md §3.3 (L3/L4) and §8.6 (improved prompt
templates with ``<analysis>`` draft blocks and priority-tagged outputs).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from prompts.prompt_loader import load_compress
from src.llm.interface import LLMInterface

# Regex that removes an <analysis>...</analysis> draft block (the L3/L4 prompt
# asks the model to draft analysis before producing the final summary; the
# draft is not stored). Flags: DOTALL so newlines match; IGNORECASE for safety
# against casing drift from different providers.
_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)


def _strip_analysis(text: str) -> str:
    """Remove any ``<analysis>...</analysis>`` block(s) from the LLM output."""
    return _ANALYSIS_RE.sub("", text).strip()


def _now_iso_z() -> str:
    """Return the current UTC time as ISO 8601 with trailing ``Z``."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _estimate_tokens(text: str) -> int:
    """Rough token count using the project's 4:1 char heuristic."""
    return max(0, len(text) // 4)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Render messages as a plain-text transcript for the compressor LLM."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "") or ""
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


async def l3_collapse(
    messages: list[dict[str, Any]],
    llm: LLMInterface,
    start_round: int,
    end_round: int,
    prompt_loader_fn: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Compress a ~20-round conversation window into one event-level block.

    Args:
        messages: Raw (L1-cleaned) messages covering rounds ``[start_round,
            end_round]`` inclusive. Each dict must carry ``role`` and
            ``content``.
        llm: An ``LLMInterface`` instance used for the single non-streaming
            summarisation call (typically the ``l3_compress`` role).
        start_round: Inclusive first round number covered by this block.
        end_round: Inclusive last round number covered by this block.
        prompt_loader_fn: Callable returning the L3 prompt template string.
            Defaults to ``load_compress('l3_collapse')``; injectable so tests
            can substitute a minimal template without touching the filesystem.

    Returns:
        Block dict with keys ``block_id`` (``block_<8hex>``), ``level`` (``0``),
        ``covers_rounds`` (``[start_round, end_round]``), ``created_at``
        (ISO 8601 UTC with ``Z``), ``summary`` (analysis-stripped text), and
        ``token_count`` (4:1 char approximation).
    """
    template = (prompt_loader_fn or (lambda: load_compress("l3_collapse")))()

    # Placeholder substitution. Use ``str.replace`` (not ``str.format``) so
    # literal braces in the template body are never misinterpreted.
    n = end_round - start_round + 1
    system_prompt = (
        template.replace("{N}", str(n))
        .replace("{start}", str(start_round))
        .replace("{end}", str(end_round))
    )

    user_payload = [{"role": "user", "content": _format_messages(messages)}]
    raw_response = await llm.chat_completion(user_payload, system=system_prompt)
    summary = _strip_analysis(raw_response)

    return {
        "block_id": f"block_{uuid.uuid4().hex[:8]}",
        "level": 0,
        "covers_rounds": [start_round, end_round],
        "created_at": _now_iso_z(),
        "summary": summary,
        "token_count": _estimate_tokens(summary),
    }


async def l4_super_compact(
    blocks: list[dict[str, Any]],
    llm: LLMInterface,
    prompt_loader_fn: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Merge several event-level blocks into one pattern-level meta-block.

    Args:
        blocks: A list of L3 blocks (level 0) to integrate. Each block must
            carry ``block_id``, ``covers_rounds`` (``[start, end]``) and
            ``summary``. Length must be ``>= 2``; the default MemoryManager
            trigger supplies 4 blocks but the function itself only requires
            at least two to form a pattern.
        llm: An ``LLMInterface`` instance (typically the ``l4_compact`` role).
        prompt_loader_fn: Callable returning the L4 prompt template string.
            Defaults to ``load_compress('l4_super_compact')``; injectable for
            tests.

    Returns:
        Meta-block dict with keys ``block_id`` (``meta_<8hex>``), ``level``
        (``1``), ``covers_rounds`` (``[first_block.start, last_block.end]``),
        ``created_at`` (ISO 8601 UTC ``Z``), ``summary`` (analysis-stripped),
        ``token_count`` (4:1 char approx), and ``source_blocks`` (input
        ``block_id`` list, preserving order).

    Raises:
        ValueError: If ``blocks`` is empty or has fewer than two entries.
    """
    if len(blocks) < 2:
        raise ValueError(f"l4_super_compact requires at least 2 blocks, got {len(blocks)}")

    template = (prompt_loader_fn or (lambda: load_compress("l4_super_compact")))()

    first_start = blocks[0]["covers_rounds"][0]
    last_end = blocks[-1]["covers_rounds"][1]
    total_rounds = last_end - first_start + 1

    def _fmt(b: dict[str, Any]) -> str:
        bid = b["block_id"]
        rs, re_ = b["covers_rounds"]
        return f"### {bid} (轮次 {rs}-{re_})\n{b['summary']}"

    block_summaries_joined = "\n\n".join(_fmt(b) for b in blocks)

    system_prompt = (
        template.replace("{N}", str(len(blocks)))
        .replace("{total_rounds}", str(total_rounds))
        .replace("{start}", str(first_start))
        .replace("{end}", str(last_end))
        .replace("{block_summaries_joined}", block_summaries_joined)
    )

    user_payload = [{"role": "user", "content": block_summaries_joined}]
    raw_response = await llm.chat_completion(user_payload, system=system_prompt)
    summary = _strip_analysis(raw_response)

    return {
        "block_id": f"meta_{uuid.uuid4().hex[:8]}",
        "level": 1,
        "covers_rounds": [first_start, last_end],
        "created_at": _now_iso_z(),
        "summary": summary,
        "token_count": _estimate_tokens(summary),
        "source_blocks": [b["block_id"] for b in blocks],
    }


__all__ = ["l3_collapse", "l4_super_compact"]
