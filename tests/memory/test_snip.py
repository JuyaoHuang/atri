"""Tests for L1 Snip pure-rule cleaner (src/memory/snip.py).

Covers all acceptance criteria from PRD US-MEM-001.
"""

from __future__ import annotations

import pytest

from src.memory.snip import snip

# Canonical filler-word list from §8.7 of the design doc.
_FILLERS = ["嗯", "啊", "呃", "额", "那个", "就是说", "对对对"]


def _cfg(
    filler_words: list[str] | None = None,
    similarity_threshold: float = 0.95,
    max_single_message_tokens: int = 800,
) -> dict:
    return {
        "filler_words": filler_words if filler_words is not None else _FILLERS,
        "similarity_threshold": similarity_threshold,
        "max_single_message_tokens": max_single_message_tokens,
    }


def test_empty_input_returns_empty() -> None:
    messages, freed = snip([], _cfg())
    assert messages == []
    assert freed == 0


def test_filler_word_removal_chinese() -> None:
    msgs = [{"role": "human", "content": "嗯那个我想问一下就是说天气"}]
    cleaned, freed = snip(msgs, _cfg())
    # "嗯", "那个", "就是说" all present in list; after removal:
    # "我想问一下天气" (spaces collapsed to nothing for adjacent-Chinese text)
    assert cleaned[0]["content"] == "我想问一下天气"
    assert freed > 0
    # Also verifies longest-first ordering: "就是说" must be removed before "就是"
    # (which isn't in the list, but this guards against future additions).


def test_adjacent_duplicate_collapse() -> None:
    # 3 consecutive similar human messages; only the last should survive.
    msgs = [
        {"role": "human", "content": "你好"},
        {"role": "human", "content": "你好"},
        {"role": "human", "content": "你好"},
    ]
    cleaned, freed = snip(msgs, _cfg())
    human_msgs = [m for m in cleaned if m["role"] == "human"]
    assert len(human_msgs) == 1
    assert human_msgs[0]["content"] == "你好"
    # freed counts the 2 dropped duplicates' full lengths.
    assert freed >= len("你好") * 2


def test_adjacent_duplicates_separated_by_ai_do_not_collapse() -> None:
    # §3.2 semantics: duplicates are "adjacent" in the message stream.
    # An AI reply between them breaks the adjacency.
    msgs = [
        {"role": "human", "content": "你好"},
        {"role": "ai", "content": "你好呀"},
        {"role": "human", "content": "你好"},
    ]
    cleaned, _freed = snip(msgs, _cfg())
    humans = [m for m in cleaned if m["role"] == "human"]
    assert len(humans) == 2


def test_overlong_message_truncated_with_marker() -> None:
    # max_tokens=10 -> max_chars=40. Provide 100 chars.
    long_content = "a" * 100
    msgs = [{"role": "human", "content": long_content}]
    cleaned, freed = snip(msgs, _cfg(max_single_message_tokens=10))
    truncated = cleaned[0]["content"]
    assert truncated.endswith("[已截断]")
    assert len(truncated) == 40 + len("[已截断]")
    # freed = original_len - truncated_len (still positive)
    assert freed == 100 - len(truncated)


def test_short_message_not_truncated() -> None:
    msgs = [{"role": "human", "content": "short"}]
    cleaned, freed = snip(msgs, _cfg(max_single_message_tokens=10))
    assert cleaned[0]["content"] == "short"
    assert freed == 0


def test_heartbeat_dropped() -> None:
    msgs = [
        {"role": "human", "content": "[heartbeat] alive"},
        {"role": "human", "content": "真正的消息"},
    ]
    cleaned, freed = snip(msgs, _cfg())
    humans = [m for m in cleaned if m["role"] == "human"]
    assert len(humans) == 1
    assert humans[0]["content"] == "真正的消息"
    # The heartbeat's full length was freed.
    assert freed >= len("[heartbeat] alive")


def test_ai_messages_pass_through_untouched() -> None:
    ai_content = "嗯那个我就是说，AI 的回复不应该被清洗"
    msgs = [
        {"role": "ai", "content": ai_content},
        {"role": "system", "content": "[heartbeat] alive"},
    ]
    cleaned, freed = snip(msgs, _cfg())
    # AI content preserved verbatim; system payload also passes through here
    # (heartbeat rule applies only to human role).
    assert cleaned[0]["content"] == ai_content
    assert cleaned[1]["content"] == "[heartbeat] alive"
    assert freed == 0


def test_tokens_freed_is_accurate_sum() -> None:
    # Carefully controlled scenario to verify the freed count.
    msgs = [
        {"role": "human", "content": "嗯嗯你好"},  # 4 chars, "嗯嗯" removed -> 2 chars, freed 2
        {"role": "ai", "content": "你好呀"},
        {"role": "human", "content": "[heartbeat] alive"},  # dropped entirely, freed 17
    ]
    _cleaned, freed = snip(msgs, _cfg())
    # Sum: (4-2) + 17 = 19
    assert freed == 19


def test_config_values_read_from_dict_not_hardcoded() -> None:
    # Use an unusual filler word that isn't in the default list.
    custom_filler = ["ZZZ"]
    msgs = [{"role": "human", "content": "ZZZhello"}]
    cleaned, _ = snip(msgs, _cfg(filler_words=custom_filler))
    assert cleaned[0]["content"] == "hello"


def test_missing_content_treated_as_empty_string() -> None:
    # Be forgiving toward malformed input (defensive).
    msgs = [{"role": "human"}]
    cleaned, freed = snip(msgs, _cfg())
    assert cleaned[0]["content"] == ""
    assert freed == 0


def test_metadata_row_passes_through() -> None:
    # chat_history.json carries a metadata row; snip must not touch it.
    meta = {"role": "metadata", "session_id": "abc", "character": "katou"}
    msgs = [meta, {"role": "human", "content": "嗯你好"}]
    cleaned, _ = snip(msgs, _cfg())
    assert cleaned[0] is meta
    assert cleaned[1]["content"] == "你好"


@pytest.mark.parametrize("filler", _FILLERS)
def test_each_canonical_filler_is_removed(filler: str) -> None:
    content = f"{filler}测试"
    msgs = [{"role": "human", "content": content}]
    cleaned, _ = snip(msgs, _cfg())
    assert filler not in cleaned[0]["content"]
    assert "测试" in cleaned[0]["content"]
