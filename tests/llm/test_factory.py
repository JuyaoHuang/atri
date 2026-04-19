"""Tests for src.llm.factory.

测试目的：验证 LLMFactory 注册机制与基于角色的工厂函数——装饰器注册不会
修改被装饰的类、能够通过 ``name`` 实例化已注册提供商、未知名称会列出可用
提供商、``available()`` 返回排序后的名称列表；同时验证 ``create_from_role``
的 happy path、缺失角色/池条目/provider 字段时的错误处理，以及不会修改传入
的配置字典。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from typing import Any

import pytest

from src.llm.factory import LLMFactory, create_from_role
from src.llm.interface import LLMInterface


@pytest.fixture(autouse=True)
def _snapshot_registry() -> Generator[None, None, None]:
    """Preserve the registry across tests so production providers stay intact."""
    before = dict(LLMFactory._registry)
    yield
    LLMFactory._registry.clear()
    LLMFactory._registry.update(before)


class _DummyLLM(LLMInterface):
    def __init__(self, model: str = "dummy", **kwargs: Any) -> None:
        self.model = model
        self.extra = kwargs

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        yield ""


def test_register_returns_decorator_that_leaves_class_unchanged() -> None:
    decorator = LLMFactory.register("dummy_a")
    result = decorator(_DummyLLM)
    assert result is _DummyLLM
    assert LLMFactory._registry["dummy_a"] is _DummyLLM


def test_register_via_decorator_syntax() -> None:
    @LLMFactory.register("dummy_b")
    class Inner(_DummyLLM):
        pass

    assert LLMFactory._registry["dummy_b"] is Inner


def test_create_instantiates_registered_class_with_kwargs() -> None:
    LLMFactory.register("dummy_c")(_DummyLLM)
    instance = LLMFactory.create("dummy_c", model="gpt-x", extra_key="v")
    assert isinstance(instance, _DummyLLM)
    assert instance.model == "gpt-x"
    assert instance.extra == {"extra_key": "v"}


def test_create_unknown_name_lists_available_providers() -> None:
    LLMFactory.register("dummy_d1")(_DummyLLM)
    LLMFactory.register("dummy_d2")(_DummyLLM)
    with pytest.raises(ValueError) as exc_info:
        LLMFactory.create("missing")
    msg = str(exc_info.value)
    assert "missing" in msg
    assert "dummy_d1" in msg
    assert "dummy_d2" in msg


def test_available_returns_sorted_names() -> None:
    LLMFactory._registry.clear()
    LLMFactory.register("zeta")(_DummyLLM)
    LLMFactory.register("alpha")(_DummyLLM)
    assert LLMFactory.available() == ["alpha", "zeta"]


def test_create_from_role_happy_path() -> None:
    LLMFactory.register("dummy_role_ok")(_DummyLLM)
    config = {
        "llm_configs": {
            "pool_x": {"provider": "dummy_role_ok", "model": "m1"},
        },
        "llm_roles": {"chat": "pool_x"},
    }
    llm = create_from_role("chat", config)
    assert isinstance(llm, _DummyLLM)
    assert llm.model == "m1"


def test_create_from_role_missing_role_raises() -> None:
    config = {"llm_configs": {}, "llm_roles": {"chat": "x"}}
    with pytest.raises(KeyError, match="not found in llm_roles"):
        create_from_role("nonexistent", config)


def test_create_from_role_missing_pool_entry_raises() -> None:
    config = {"llm_configs": {}, "llm_roles": {"chat": "missing_key"}}
    with pytest.raises(KeyError, match="missing from llm_configs"):
        create_from_role("chat", config)


def test_create_from_role_missing_provider_raises() -> None:
    config = {
        "llm_configs": {"pool_y": {"model": "m"}},
        "llm_roles": {"chat": "pool_y"},
    }
    with pytest.raises(KeyError, match="'provider'"):
        create_from_role("chat", config)


def test_create_from_role_does_not_mutate_input_config() -> None:
    LLMFactory.register("dummy_role_imm")(_DummyLLM)
    entry = {"provider": "dummy_role_imm", "model": "m"}
    config = {"llm_configs": {"p": entry}, "llm_roles": {"chat": "p"}}
    create_from_role("chat", config)
    assert entry == {"provider": "dummy_role_imm", "model": "m"}
