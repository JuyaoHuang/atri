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

根``config.yaml``包含如下条目：

    llm_config：config/llm_config.yaml
    内存配置：config/memory_config.yaml

该加载器读取根目录，解析相对于根目录的每个子路径
文件的目录，并将子内容内联到由以下键控的合并字典中
剥离的主干（``llm_config`` -> ``llm``）。 ``${ENV_VAR}`` 标记是
在合并期间从“os.environ”替换；未设置的变量保留
他们的文字占位符，因此未使用的配置分支不会强迫用户
填充每个引用的变量。

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
