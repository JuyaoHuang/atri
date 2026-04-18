"""Application entry point.

Initializes the logger, loads the root config, then resolves each of the
three configured LLM call-sites (``chat``, ``l3_compress``, ``l4_compact``)
via the role-based factory. Instantiating the LLM clients does not touch
the network -- only construction; actual requests happen later in the
agent/memory layers.

Run::

    uv run python -m src.main

应用程序入口点。

初始化日志记录器，加载根配置，然后通过基于角色的工厂解析三个已配置的
LLM 调用位点 (``chat``、``l3_compress``、``l4_compact``) 中的每一个。
实例化 LLM 客户端不会触发网络请求——仅执行构造；实际请求稍后在
agent/memory 层中发生。

运行方式::

    uv run python -m src.main

Reference: docs/项目架构设计.md §2.2, docs/LLM调用层设计讨论.md §2.3
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.llm import create_from_role
from src.utils.config_loader import load_config
from src.utils.logger import init_logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"

_LLM_ROLES = ("chat", "l3_compress", "l4_compact")


def main() -> None:
    init_logger()
    logger.info("atri starting")

    config = load_config(_DEFAULT_CONFIG)
    logger.info("Config loaded | sections={}", sorted(config.keys()))

    llm_config = config.get("llm", {})
    for role in _LLM_ROLES:
        try:
            llm = create_from_role(role, llm_config)
        except (KeyError, ValueError) as exc:
            logger.error("LLM role failed | role={} error={}", role, exc)
            continue
        logger.info(
            "LLM role resolved | role={} provider={} model={}",
            role,
            type(llm).__name__,
            getattr(llm, "model", "n/a"),
        )


if __name__ == "__main__":
    main()
