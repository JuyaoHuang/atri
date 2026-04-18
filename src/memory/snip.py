"""L1 Snip -- pure-rule cleaning for human messages.

Applies filler-word removal, adjacent-duplicate collapse, overlong truncation
and heartbeat/system payload drop. Never mutates ``role=="ai"`` or
``role=="system"`` messages.

Reference: docs/记忆系统设计讨论.md §3.3 (L1) and §8.7 (curated filler-word
list plus the "prefer under-delete over wrong-delete" principle).

Token approximation: ``len(content) // 4``. This is an ASCII-biased rough
heuristic used only for truncation thresholds and ``tokens_freed`` bookkeeping;
it is never persisted. For Chinese-heavy content the approximation
under-counts tokens, which pairs well with §8.7's conservative stance.

L1 Snip——针对人类消息的纯规则清洗层。

执行以下操作：去除填充词、折叠相邻重复消息、超长消息截断，以及丢弃
heartbeat/系统载荷。从不修改 ``role=="ai"`` 或 ``role=="system"`` 的消息。

参考：docs/记忆系统设计讨论.md §3.3 (L1) 与 §8.7（精选的填充词列表，
以及"宁可漏删、不可错删"的原则）。

Token 近似估算：``len(content) // 4``。这是一种偏向 ASCII 的粗略启发式，
仅用于截断阈值判断和 ``tokens_freed`` 的记账；从不被持久化。对于中文为主
的内容，该估算会低估 token 数，这与 §8.7 的保守立场相契合。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_TRUNCATION_MARKER = "[已截断]"
_HEARTBEAT_RE = re.compile(r"^\s*\[heartbeat\]", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _estimate_tokens(text: str) -> int:
    """Rough token count = ``len(text) // 4``, clamped to ``>= 0``.

    粗略的 token 数估算 = ``len(text) // 4``，下限为 ``0``。
    """
    return max(0, len(text) // 4)


def _is_heartbeat(content: str) -> bool:
    """Return True when content is a heartbeat/status payload.

    当内容是 heartbeat/状态载荷时返回 True。
    """
    return bool(_HEARTBEAT_RE.match(content))


def _remove_filler_words(content: str, filler_words: list[str]) -> str:
    """Remove each filler phrase as a literal substring.

    Chinese has no word-boundary concept, so we rely on §8.7's "prefer
    under-delete over wrong-delete" principle: the filler_words list is
    curated to only contain tokens whose literal deletion is safe in any
    context. Longer phrases are removed first so that e.g. ``就是说`` is
    stripped before any shorter prefix would match.

    将每个填充短语作为字面子串删除。

    中文没有词边界概念，因此我们依赖 §8.7 的"宁可漏删、不可错删"原则：
    filler_words 列表经过精心挑选，仅包含在任意上下文中字面删除都安全的
    词条。较长的短语优先删除，例如 ``就是说`` 会在任何更短前缀匹配之前被
    剥离。
    """
    if not filler_words:
        return content
    for word in sorted(filler_words, key=len, reverse=True):
        if word:
            content = content.replace(word, "")
    # Collapse runs of whitespace caused by deletions; strip edges.
    # 合并因删除产生的连续空白；去除首尾空白。
    return _WHITESPACE_RE.sub(" ", content).strip()


def _truncate(content: str, max_tokens: int) -> str:
    """Truncate to approximately ``max_tokens`` via a 4:1 char heuristic.

    通过 4:1 字符启发式截断到大约 ``max_tokens`` 的长度。
    """
    max_chars = max(0, max_tokens) * 4
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + _TRUNCATION_MARKER


def _similar(a: str, b: str, threshold: float) -> bool:
    """Return True when the two strings' similarity ratio meets ``threshold``.

    当两个字符串的相似度比值达到 ``threshold`` 时返回 True。
    """
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def snip(
    messages: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Apply L1 rule-based cleaning to ``role=="human"`` messages.

    Args:
        messages: Input list; each entry should have ``role`` and ``content``.
            Non-human roles pass through untouched.
        config: Dict with ``filler_words`` (list[str]),
            ``similarity_threshold`` (float in ``[0, 1]``),
            ``max_single_message_tokens`` (int).

    Returns:
        ``(cleaned_messages, tokens_freed)`` where ``tokens_freed`` is the
        non-negative sum of ``len(original) - len(new)`` across every mutated
        human message. Dropped or collapsed messages contribute their full
        original length.

    对 ``role=="human"`` 的消息应用 L1 基于规则的清洗。

    参数：
        messages：输入列表；每个条目应包含 ``role`` 和 ``content``。
            非 human 角色原样透传，不做修改。
        config：包含 ``filler_words``（list[str]）、
            ``similarity_threshold``（``[0, 1]`` 区间内的 float）、
            ``max_single_message_tokens``（int）的字典。

    返回：
        ``(cleaned_messages, tokens_freed)``，其中 ``tokens_freed`` 是所有
        被修改的 human 消息 ``len(原始) - len(新)`` 的非负和。被丢弃或折叠
        的消息贡献其完整的原始长度。
    """
    filler_words: list[str] = list(config.get("filler_words", []))
    similarity_threshold = float(config.get("similarity_threshold", 0.95))
    max_tokens = int(config.get("max_single_message_tokens", 800))

    cleaned: list[dict[str, Any]] = []
    tokens_freed = 0

    for msg in messages:
        role = msg.get("role")
        if role != "human":
            cleaned.append(msg)
            continue

        content = msg.get("content", "") or ""
        original_len = len(content)

        # Drop heartbeat/status payloads entirely.
        # 整条丢弃 heartbeat/状态载荷。
        if _is_heartbeat(content):
            tokens_freed += original_len
            continue

        # Adjacent-duplicate collapse: if the previous kept message is a
        # human message similar enough, drop *that* previous one and keep
        # the current (§3.3 L1 "保留最后一条").
        # 相邻重复折叠：若已保留的上一条是相似度足够的 human 消息，则丢弃
        # *上一条* 并保留当前条（§3.3 L1 "保留最后一条"）。
        if cleaned and cleaned[-1].get("role") == "human":
            prev_content = cleaned[-1].get("content", "") or ""
            if _similar(content, prev_content, similarity_threshold):
                tokens_freed += len(prev_content)
                cleaned.pop()

        # Filler-word removal, then hard truncation.
        # 先删除填充词，再做硬截断。
        stripped = _remove_filler_words(content, filler_words)
        truncated = _truncate(stripped, max_tokens)

        delta = original_len - len(truncated)
        if delta > 0:
            tokens_freed += delta

        new_msg = dict(msg)
        new_msg["content"] = truncated
        cleaned.append(new_msg)

    return cleaned, max(0, tokens_freed)


__all__ = ["snip"]
