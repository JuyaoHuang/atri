"""LLM call-layer exception hierarchy.

The interface layer only raises exceptions -- retry policy is decided by the
caller (§2.6). Provider implementations map SDK-specific errors to these
subclasses so downstream code can switch on exception type without coupling
to any one SDK.

LLM 调用层异常层次结构。

接口层只负责抛出异常——重试策略由调用方决定 (§2.6)。提供商实现将 SDK
特定的错误映射到这些子类，这样下游代码可以基于异常类型进行分支处理，
而无需耦合到任何特定 SDK。

Reference: docs/LLM调用层设计讨论.md §2.6
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM call-layer errors.

    所有 LLM 调用层错误的基类。
    """


class LLMConnectionError(LLMError):
    """Cannot reach the LLM service (DNS, TCP, TLS, timeout).

    无法连接到 LLM 服务（DNS、TCP、TLS、超时）。
    """


class LLMRateLimitError(LLMError):
    """Provider rate limit or quota has been exceeded.

    提供商的速率限制或配额已超出。
    """


class LLMAPIError(LLMError):
    """API-level error not covered by the more specific subclasses.

    未被更具体的子类覆盖的 API 级别错误。
    """
