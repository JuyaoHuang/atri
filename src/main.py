"""Application entry point.

Initializes the logger, loads the root config, then resolves each of the
three configured LLM call-sites (``chat``, ``l3_compress``, ``l4_compact``)
via the role-based factory, constructs a :class:`ServiceContext`, and
materializes a single ChatAgent for the ``atri`` character as a wiring
smoke test. **No network requests happen** -- we only construct clients
and verify role resolution; the actual LLM / mem0 calls happen later in
the agent/memory layers during a real conversation.

Startup sequence:

1. ``init_logger()`` / ``load_dotenv()``
2. ``load_config()`` on ``config.yaml``
3. Resolve 3 LLM roles (chat / l3_compress / l4_compact) for logging
4. Construct :class:`ServiceContext` (holds config, empty agent cache)
5. ``ctx.get_or_create_agent("atri", user_id="main_demo")`` to build one
   ChatAgent end-to-end (Persona + LongTermMemory + MemoryManager + LLM)
6. Log ``ChatAgent ready | character=atri | persona={name} | long_term={on|off}``

Run::

    uv run python -m src.main

应用程序入口点。

初始化日志记录器，加载根配置，通过基于角色的工厂解析三个已配置的 LLM 调用
位点（``chat`` / ``l3_compress`` / ``l4_compact``），构造
:class:`ServiceContext`，并为 ``atri`` 角色物化一个 ChatAgent 作为接线
冒烟测试。**不触发任何网络请求**——只构造客户端并验证角色解析；真正的
LLM / mem0 调用稍后在真实对话中由 agent/memory 层发起。

Reference: docs/项目架构设计.md §2.2, §2.5,
docs/LLM调用层设计讨论.md §2.3,
docs/记忆系统设计讨论.md §6.1,
docs/Phase4_执行规格.md §US-AGT-006
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.llm import create_from_role
from src.service_context import ServiceContext
from src.utils.config_loader import load_config
from src.utils.logger import init_logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"
_DEFAULT_ENV = _REPO_ROOT / ".env"

_LLM_ROLES = ("chat", "l3_compress", "l4_compact")
_DEMO_CHARACTER = "atri"
_DEMO_USER_ID = "main_demo"


def main() -> None:
    init_logger()
    logger.info("atri starting")

    dotenv_loaded = load_dotenv(_DEFAULT_ENV)
    logger.info("Dotenv loaded | path={} present={}", _DEFAULT_ENV, dotenv_loaded)

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

    try:
        ctx = ServiceContext(config)
        agent = ctx.get_or_create_agent(_DEMO_CHARACTER, user_id=_DEMO_USER_ID)
    except Exception as exc:  # noqa: BLE001
        logger.error("ServiceContext / ChatAgent construction failed | error={!r}", exc)
        return

    mgr = agent.memory_manager
    mem0_mode = config.get("memory", {}).get("mem0", {}).get("mode", "local_deploy")
    logger.info(
        "MemoryManager ready | mode={} | character={} | long_term={}",
        mem0_mode,
        _DEMO_CHARACTER,
        "on" if mgr.long_term is not None else "off",
    )
    logger.info(
        "ChatAgent ready | character={} | persona={} | long_term={}",
        _DEMO_CHARACTER,
        agent.persona.name,
        "on" if mgr.long_term is not None else "off",
    )


if __name__ == "__main__":
    main()
