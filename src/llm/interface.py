"""Abstract LLM interface -- stateless streaming + non-streaming.

Stateless by design: system prompt and messages are supplied on every call,
never stored on the instance. Subclasses implement only the streaming
method; the non-streaming variant has a default implementation that
collects the stream, so concrete providers only carry one abstract method.

The ``tools`` parameter is reserved for future tool-calling support and is
currently ignored by all providers.

抽象 LLM 接口——无状态的流式 + 非流式。

设计上无状态：系统提示和消息在每次调用时提供，从不存储在实例上。子类只需
实现流式方法；非流式变体有一个收集流的默认实现，因此具体的提供商只需要
实现一个抽象方法。

``tools`` 参数是为将来的工具调用支持保留的，目前所有提供商都会忽略它。

Reference: docs/LLM调用层设计讨论.md §2.2, §2.7
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMInterface(ABC):
    """Stateless LLM call contract.

    Subclasses must implement :meth:`chat_completion_stream` as an async
    generator. :meth:`chat_completion` defaults to collecting the stream.

    无状态的 LLM 调用契约。

    子类必须将 :meth:`chat_completion_stream` 实现为异步生成器。
    :meth:`chat_completion` 默认通过收集流式输出实现。
    """

    @abstractmethod
    def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield non-empty content deltas from the LLM response stream.

        Args:
            messages: Chat history in OpenAI-style format -- a list of
                ``{"role": "user" | "assistant" | ..., "content": str}``.
            system: Optional system prompt. Providers prepend this as a
                ``{"role": "system", "content": system}`` message when
                present.
            tools: Reserved for tool-calling (§2.7). Currently unused by
                all providers.

        Yields:
            str: Content chunks in order of arrival. Empty chunks are
            skipped by provider implementations.

        从 LLM 响应流中产出非空的内容增量。

        参数：
            messages：OpenAI 风格格式的聊天历史——一个
                ``{"role": "user" | "assistant" | ..., "content": str}`` 的列表。
            system：可选的系统提示。若存在，提供商会将其作为
                ``{"role": "system", "content": system}`` 消息前置。
            tools：保留用于工具调用 (§2.7)。目前所有提供商均未使用。

        产出：
            str：按到达顺序排列的内容块。空块会被提供商实现跳过。
        """

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return the full response as a single string.

        Default implementation collects the streaming result. Override
        when a non-streaming API path is meaningfully cheaper.

        以单个字符串形式返回完整响应。

        默认实现通过收集流式结果实现。当非流式 API 路径开销明显更低时，
        子类可以覆盖此方法。
        """
        parts: list[str] = []
        async for chunk in self.chat_completion_stream(messages, system, tools):
            parts.append(chunk)
        return "".join(parts)
