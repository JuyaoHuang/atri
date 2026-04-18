"""OpenAI-compatible LLM provider.

Uses the ``openai`` SDK's ``AsyncOpenAI`` client with a configurable
``base_url``. This one provider covers OpenAI, DeepSeek, Groq, Mistral,
vLLM, LM Studio, and Ollama's ``/v1`` endpoint -- any service that speaks
the OpenAI chat-completions protocol.

Exceptions from the SDK are translated into the project-local hierarchy
(:mod:`src.llm.exceptions`) so callers never need to import ``openai``.

OpenAI 兼容的 LLM 提供商。

使用 ``openai`` SDK 的 ``AsyncOpenAI`` 客户端，搭配可配置的 ``base_url``。
这一个提供商即可覆盖 OpenAI、DeepSeek、Groq、Mistral、vLLM、LM Studio
以及 Ollama 的 ``/v1`` 端点——任何使用 OpenAI chat-completions 协议的服务。

SDK 抛出的异常会被转换为项目本地的异常层次结构
（:mod:`src.llm.exceptions`），调用方无需导入 ``openai``。

Reference: docs/LLM调用层设计讨论.md §2.4, §2.6
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    RateLimitError,
)

from src.llm.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMRateLimitError,
)
from src.llm.factory import LLMFactory
from src.llm.interface import LLMInterface


@LLMFactory.register("openai_compatible")
class OpenAICompatibleLLM(LLMInterface):
    """LLM provider speaking the OpenAI chat-completions protocol.

    使用 OpenAI chat-completions 协议的 LLM 提供商。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float | None = None,
        **extra: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.extra = extra
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload: list[dict[str, Any]] = (
            [{"role": "system", "content": system}, *messages]
            if system is not None
            else list(messages)
        )

        params: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "stream": True,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APIConnectionError as exc:
            raise LLMConnectionError(str(exc)) from exc
        except RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except APIError as exc:
            raise LLMAPIError(str(exc)) from exc
        except Exception as exc:
            raise LLMAPIError(str(exc)) from exc
