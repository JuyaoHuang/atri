"""OpenAI-compatible LLM provider.

Uses the ``openai`` SDK's ``AsyncOpenAI`` client with a configurable
``base_url``. This one provider covers OpenAI, DeepSeek, Groq, Mistral,
vLLM, LM Studio, and Ollama's ``/v1`` endpoint -- any service that speaks
the OpenAI chat-completions protocol.

Exceptions from the SDK are translated into the project-local hierarchy
(:mod:`src.llm.exceptions`) so callers never need to import ``openai``.

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
    """LLM provider speaking the OpenAI chat-completions protocol."""

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
