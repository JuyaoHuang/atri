"""Tests for src/memory/compressor.py -- L3 Collapse.

All acceptance criteria from PRD US-MEM-002 covered here.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.memory.compressor import l3_collapse

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MINIMAL_TEMPLATE = (
    "SYS: compress {N} rounds ({start}-{end}).\n"
    "<analysis>instructions not stored</analysis>\n"
    "Write the summary below."
)


def _fake_loader(template: str = _MINIMAL_TEMPLATE):
    return lambda: template


def _mock_llm(response_text: str) -> Any:
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response_text)
    return llm


_SAMPLE_MESSAGES = [
    {"role": "human", "content": "嗯你好"},
    {"role": "ai", "content": "你好呀"},
    {"role": "human", "content": "今天天气怎么样"},
    {"role": "ai", "content": "挺不错的。"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_called_exactly_once() -> None:
    llm = _mock_llm("<analysis>thinking</analysis>## final summary\n- a\n- b")
    await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20, prompt_loader_fn=_fake_loader())
    assert llm.chat_completion.await_count == 1


@pytest.mark.asyncio
async def test_analysis_block_stripped_including_nested_newlines() -> None:
    raw = (
        "<analysis>\n"
        "line 1\n"
        "line 2 with special chars 嗯啊\n"
        "</analysis>\n"
        "## 对话摘要 (轮次 1-20)\n"
        "- 关键事实: test\n"
    )
    llm = _mock_llm(raw)
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20, prompt_loader_fn=_fake_loader())
    assert "<analysis>" not in block["summary"]
    assert "</analysis>" not in block["summary"]
    assert "对话摘要" in block["summary"]
    assert "关键事实: test" in block["summary"]


@pytest.mark.asyncio
async def test_multiple_analysis_blocks_all_stripped() -> None:
    raw = "<analysis>first</analysis>A<analysis>second</analysis>B"
    llm = _mock_llm(raw)
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20, prompt_loader_fn=_fake_loader())
    assert "<analysis>" not in block["summary"]
    assert "A" in block["summary"]
    assert "B" in block["summary"]


@pytest.mark.asyncio
async def test_block_id_shape() -> None:
    llm = _mock_llm("plain summary")
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20, prompt_loader_fn=_fake_loader())
    assert re.fullmatch(r"block_[0-9a-f]{8}", block["block_id"])


@pytest.mark.asyncio
async def test_covers_rounds_matches_args() -> None:
    llm = _mock_llm("summary")
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 41, 60, prompt_loader_fn=_fake_loader())
    assert block["covers_rounds"] == [41, 60]


@pytest.mark.asyncio
async def test_created_at_is_iso_with_z() -> None:
    from datetime import datetime

    llm = _mock_llm("summary")
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20, prompt_loader_fn=_fake_loader())
    ts = block["created_at"]
    assert ts.endswith("Z")
    # Strip Z and parse to confirm it is a valid ISO 8601 timestamp.
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


@pytest.mark.asyncio
async def test_summary_contains_non_analysis_portion_verbatim() -> None:
    tail = "## 对话摘要 (轮次 1-20)\n- 情感轨迹: 平静\n- 关键事实: 用户喜欢猫"
    raw = f"<analysis>draft</analysis>\n{tail}"
    llm = _mock_llm(raw)
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20, prompt_loader_fn=_fake_loader())
    # The tail survives intact (whitespace trimming on the outer ends is OK).
    assert block["summary"] == tail.strip()


@pytest.mark.asyncio
async def test_placeholders_substituted_in_prompt() -> None:
    llm = _mock_llm("ok")
    await l3_collapse(_SAMPLE_MESSAGES, llm, 41, 60, prompt_loader_fn=_fake_loader())
    # Inspect the system prompt actually passed to the LLM.
    system = llm.chat_completion.call_args.kwargs["system"]
    assert "compress 20 rounds (41-60)" in system
    assert "{N}" not in system
    assert "{start}" not in system
    assert "{end}" not in system


@pytest.mark.asyncio
async def test_token_count_is_char_approximation() -> None:
    llm = _mock_llm("x" * 400)  # 400 chars -> ~100 tokens
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20, prompt_loader_fn=_fake_loader())
    assert block["token_count"] == 100


@pytest.mark.asyncio
async def test_default_loader_used_when_none_supplied() -> None:
    """Without an injected loader, the real prompts/compress/l3_collapse.txt is read."""
    llm = _mock_llm("summary")
    block = await l3_collapse(_SAMPLE_MESSAGES, llm, 1, 20)
    # Just check the call happened and resulting block is well formed -- the
    # placeholder substitution already guards the template path.
    assert block["summary"] == "summary"
