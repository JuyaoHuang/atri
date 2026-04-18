"""Tests for src.llm.exceptions."""

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
