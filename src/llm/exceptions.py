"""LLM call-layer exception hierarchy.

The interface layer only raises exceptions -- retry policy is decided by the
caller (§2.6). Provider implementations map SDK-specific errors to these
subclasses so downstream code can switch on exception type without coupling
to any one SDK.

Reference: docs/LLM调用层设计讨论.md §2.6
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM call-layer errors."""


class LLMConnectionError(LLMError):
    """Cannot reach the LLM service (DNS, TCP, TLS, timeout)."""


class LLMRateLimitError(LLMError):
    """Provider rate limit or quota has been exceeded."""


class LLMAPIError(LLMError):
    """API-level error not covered by the more specific subclasses."""
