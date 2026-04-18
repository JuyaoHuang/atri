"""Layered config loader -- root config references sub-files by path.

The root ``config.yaml`` contains entries like::

    llm_config: config/llm_config.yaml
    memory_config: config/memory_config.yaml

This loader reads the root, resolves each sub-path relative to the root
file's directory, and inlines the sub-content into a merged dict keyed by
the stripped stem (``llm_config`` -> ``llm``). ``${ENV_VAR}`` tokens are
substituted from ``os.environ`` during the merge; unset variables keep
their literal placeholder so unused config branches don't force the user
to populate every referenced variable.

分层配置加载器——根配置按路径引用子文件。

根 ``config.yaml`` 包含如下条目::

    llm_config: config/llm_config.yaml
    memory_config: config/memory_config.yaml

该加载器读取根文件，将每个子路径解析为相对于根文件所在目录的路径，并将子
内容内联到一个合并字典中，键名为去掉后缀 ``_config`` 的名称
（``llm_config`` -> ``llm``）。合并期间 ``${ENV_VAR}`` 标记会从
``os.environ`` 中替换；未设置的变量保留其字面占位符，这样未使用的配置分支
不会强制用户填充每个被引用的变量。

Reference: docs/LLM调用层设计讨论.md §2.5
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")
_CONFIG_SUFFIX = "_config"


def _substitute_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    return value


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(root_path: str | Path) -> dict[str, Any]:
    """Load root config + referenced sub-config files, env-substituted.

    Any root key whose name ends in ``_config`` and maps to a string is
    treated as a path reference. The referenced YAML is read and inlined
    under the stripped stem. Other root keys are copied verbatim. Paths
    are resolved relative to the root file's parent directory.

    Args:
        root_path: Path to the root ``config.yaml``.

    Returns:
        Merged dict, e.g. ``{"llm": {...}, "memory": {...}}``.

    Raises:
        FileNotFoundError: Root or any referenced sub-file is missing.

    加载根配置 + 引用的子配置文件，并进行环境变量替换。

    任何名称以 ``_config`` 结尾且值为字符串的根键都会被视为路径引用。
    被引用的 YAML 文件会被读取并内联到去掉 ``_config`` 后缀的键名下。
    其他根键按原样复制。路径相对于根文件的父目录进行解析。

    参数：
        root_path：根 ``config.yaml`` 的路径。

    返回：
        合并后的字典，例如 ``{"llm": {...}, "memory": {...}}``。

    抛出：
        FileNotFoundError：根文件或任何被引用的子文件缺失。
    """
    root_file = Path(root_path).resolve()
    if not root_file.is_file():
        raise FileNotFoundError(f"Root config not found: {root_file}")

    root_data = _read_yaml(root_file) or {}
    base_dir = root_file.parent
    merged: dict[str, Any] = {}

    for key, value in root_data.items():
        if key.endswith(_CONFIG_SUFFIX) and isinstance(value, str):
            sub_path = (base_dir / value).resolve()
            if not sub_path.is_file():
                raise FileNotFoundError(f"Sub-config '{key}' references missing file: {sub_path}")
            stem = key[: -len(_CONFIG_SUFFIX)]
            merged[stem] = _substitute_env_vars(_read_yaml(sub_path) or {})
        else:
            merged[key] = _substitute_env_vars(value)

    return merged
