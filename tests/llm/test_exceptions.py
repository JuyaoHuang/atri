"""Tests for src.llm.exceptions.

测试目的：验证 LLM 调用层的异常层次结构——``LLMError`` 继承自 ``Exception``，
``LLMConnectionError``、``LLMRateLimitError`` 与 ``LLMAPIError`` 均继承自
``LLMError``，并且所有四个异常类都可以正常导入和实例化。
"""

from __future__ import annotations

import pytest

from src.llm.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
)


def test_base_is_exception_subclass() -> None:
    assert issubclass(LLMError, Exception)


@pytest.mark.parametrize(
    "subclass",
    [LLMConnectionError, LLMRateLimitError, LLMAPIError],
)
def test_subclasses_inherit_from_base(subclass: type[LLMError]) -> None:
    assert issubclass(subclass, LLMError)
    instance = subclass("boom")
    assert isinstance(instance, LLMError)
    assert isinstance(instance, Exception)
    assert str(instance) == "boom"


def test_all_four_classes_importable() -> None:
    # Smoke test for __all__-equivalent surface: all 4 names resolve at import time.
    for cls in (LLMError, LLMConnectionError, LLMRateLimitError, LLMAPIError):
        assert isinstance(cls, type)
