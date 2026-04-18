"""LLM call layer -- interfaces, factory, and built-in provider registrations.

Importing this package triggers the side-effect registration of all built-in
providers (via :mod:`src.llm.providers`), so downstream code can call
:func:`create_from_role` or :meth:`LLMFactory.create` without manually
importing provider modules.

LLM 调用层——接口、工厂和内置提供商注册。

导入此包会触发所有内置提供商的副作用注册（通过 :mod:`src.llm.providers`），
因此下游代码可以调用 :func:`create_from_role` 或 :meth:`LLMFactory.create`
而无需手动导入提供商模块。
"""

from __future__ import annotations

from src.llm.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
)
from src.llm.factory import LLMFactory, create_from_role
from src.llm.interface import LLMInterface
from src.llm.providers import openai_compatible as _openai_compatible  # noqa: F401

__all__ = [
    "LLMAPIError",
    "LLMConnectionError",
    "LLMError",
    "LLMFactory",
    "LLMInterface",
    "LLMRateLimitError",
    "create_from_role",
]
