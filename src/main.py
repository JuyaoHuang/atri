"""Application entry point - FastAPI server with uvicorn.

Initializes the logger, loads the root config, resolves LLM roles for logging,
constructs a :class:`ServiceContext`, materializes a ChatAgent for smoke test,
then starts the FastAPI server with uvicorn.

Startup sequence:

1. ``init_logger()`` / ``load_dotenv()``
2. ``load_config()`` on ``config.yaml``
3. Resolve 3 LLM roles (chat / l3_compress / l4_compact) for logging
4. Construct :class:`ServiceContext` (holds config, empty agent cache)
5. ``ctx.get_or_create_agent("atri", user_id="main_demo")`` to build one
   ChatAgent end-to-end (Persona + LongTermMemory + MemoryManager + LLM)
6. Log ``ChatAgent ready | character=atri | persona={name} | long_term={on|off}``
7. Create FastAPI app via ``create_app(config)``
8. Start uvicorn server on configured host/port

Run::

    uv run python -m src.main

应用程序入口点 - 使用 uvicorn 的 FastAPI 服务器。

初始化日志记录器，加载根配置，解析 LLM 角色用于日志记录，构造
:class:`ServiceContext`，物化一个 ChatAgent 用于冒烟测试，然后使用 uvicorn
启动 FastAPI 服务器。

Reference: docs/项目架构设计.md §2.2, §2.5,
docs/Phase5_执行规格.md §US-SRV-007
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from loguru import logger

from src.app import create_app
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

    # Resolve LLM roles for logging (smoke test)
    # 解析 LLM 角色用于日志记录（冒烟测试）
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

    # Construct ServiceContext and ChatAgent for smoke test
    # 构造 ServiceContext 和 ChatAgent 用于冒烟测试
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

    # Create FastAPI app
    # 创建 FastAPI 应用
    app = create_app(config)

    # Get server config
    # 获取服务器配置
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)

    logger.info("Server starting | host={} | port={}", host, port)

    # Start uvicorn server
    # 启动 uvicorn 服务器
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
