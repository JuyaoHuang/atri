"""LLMFactory -- decorator-based registry for pluggable providers.

Providers register themselves at import time via
``@LLMFactory.register("<name>")``. The factory code itself never changes
when a new provider is added; only the provider file + a yaml entry.

LLMFactory——基于装饰器的可插拔提供商注册表。

提供商在导入时通过 ``@LLMFactory.register("<name>")`` 自行注册。新增提供商
时，工厂代码本身不会改变；只需要新增提供商文件 + 一条 yaml 条目。

Reference: docs/LLM调用层设计讨论.md §2.1, §2.3
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.llm.interface import LLMInterface


class LLMFactory:
    """Class-scoped registry mapping provider name -> provider class.

    类作用域的注册表，将提供商名称映射到提供商类。
    """

    _registry: dict[str, type[LLMInterface]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[LLMInterface]], type[LLMInterface]]:
        """Return a decorator that binds ``name`` to the decorated class.

        返回一个装饰器，将 ``name`` 绑定到被装饰的类上。
        """

        def wrapper(llm_class: type[LLMInterface]) -> type[LLMInterface]:
            cls._registry[name] = llm_class
            return llm_class

        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> LLMInterface:
        """Instantiate a registered provider by ``name``.

        根据 ``name`` 实例化一个已注册的提供商。
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(f"Unknown LLM provider: {name!r}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return the sorted list of currently registered provider names.

        返回当前已注册的提供商名称的排序列表。
        """
        return sorted(cls._registry.keys())


def create_from_role(role: str, llm_config: dict[str, Any]) -> LLMInterface:
    """Create an LLM instance by resolving a role name through the config.

    Looks up ``llm_config['llm_roles'][role]`` to get a pool key, then
    fetches ``llm_config['llm_configs'][pool_key]`` to get the provider
    parameters, and finally calls :meth:`LLMFactory.create`.

    Args:
        role: Call-site name -- one of ``"chat" | "l3_compress" | "l4_compact"``
            (or any key defined in ``llm_roles``).
        llm_config: The ``llm`` section from the merged config (contains
            ``llm_configs`` and ``llm_roles``).

    Returns:
        A ready-to-use :class:`LLMInterface` instance.

    Raises:
        KeyError: Role not in ``llm_roles`` or referenced pool key not in
            ``llm_configs``.
        ValueError: Pool entry's ``provider`` field is not registered.

    通过配置解析角色名来创建一个 LLM 实例。

    查找 ``llm_config['llm_roles'][role]`` 得到一个池键，然后获取
    ``llm_config['llm_configs'][pool_key]`` 得到提供商参数，最终调用
    :meth:`LLMFactory.create`。

    参数：
        role：调用位点名称——``"chat" | "l3_compress" | "l4_compact"`` 之一
            （或 ``llm_roles`` 中定义的任意键）。
        llm_config：合并配置中的 ``llm`` 部分（包含 ``llm_configs`` 和
            ``llm_roles``）。

    返回：
        一个即可用的 :class:`LLMInterface` 实例。

    抛出：
        KeyError：角色不在 ``llm_roles`` 中，或引用的池键不在 ``llm_configs`` 中。
        ValueError：池条目的 ``provider`` 字段未注册。
    """
    roles = llm_config.get("llm_roles", {})
    if role not in roles:
        raise KeyError(f"Role {role!r} not found in llm_roles. Available: {sorted(roles)}")

    pool_key = roles[role]
    pool = llm_config.get("llm_configs", {})
    if pool_key not in pool:
        raise KeyError(
            f"Role {role!r} references pool key {pool_key!r}, "
            f"but it is missing from llm_configs. Available: {sorted(pool)}"
        )

    entry = dict(pool[pool_key])
    provider = entry.pop("provider", None)
    if provider is None:
        raise KeyError(f"Pool entry {pool_key!r} is missing required 'provider' field")

    return LLMFactory.create(provider, **entry)
