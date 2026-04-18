"""Tests for src.llm.interface.

测试目的：验证抽象 LLM 接口的契约——不能直接实例化抽象类、
``chat_completion`` 默认实现能正确收集多块流、能处理空流、可选参数
（``system``、``tools``）能正确透传到流式方法，以及流式输出按到达顺序产出。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from src.llm.interface import LLMInterface


class _FakeLLM(LLMInterface):
    """Minimal concrete subclass -- yields whatever chunks it was given."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.seen_system: str | None = None
        self.seen_tools: list[dict[str, Any]] | None = None
        self.seen_messages: list[dict[str, Any]] | None = None

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        self.seen_messages = messages
        self.seen_system = system
        self.seen_tools = tools
        for chunk in self.chunks:
            yield chunk


def test_cannot_instantiate_abstract_interface() -> None:
    with pytest.raises(TypeError):
        LLMInterface()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_default_chat_completion_collects_multi_chunk_stream() -> None:
    llm = _FakeLLM(["hel", "lo ", "world"])
    result = await llm.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert result == "hello world"


@pytest.mark.asyncio
async def test_default_chat_completion_handles_empty_stream() -> None:
    llm = _FakeLLM([])
    result = await llm.chat_completion(messages=[])
    assert result == ""


@pytest.mark.asyncio
async def test_optional_params_propagate_to_stream() -> None:
    llm = _FakeLLM(["x"])
    await llm.chat_completion(
        messages=[{"role": "user", "content": "q"}],
        system="you are a bot",
        tools=[{"name": "t"}],
    )
    assert llm.seen_system == "you are a bot"
    assert llm.seen_tools == [{"name": "t"}]
    assert llm.seen_messages == [{"role": "user", "content": "q"}]


@pytest.mark.asyncio
async def test_stream_yields_chunks_in_order() -> None:
    llm = _FakeLLM(["a", "b", "c"])
    seen: list[str] = []
    async for chunk in llm.chat_completion_stream(messages=[]):
        seen.append(chunk)
    assert seen == ["a", "b", "c"]
