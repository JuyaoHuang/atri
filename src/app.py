"""FastAPI application factory and lifespan management.

Creates the FastAPI application instance with:

1. **Lifespan management**: Initialize ServiceContext and ChatStorage on startup,
   cleanup on shutdown via ``close_all()``
2. **CORS middleware**: Configure allowed origins from ``server_config.cors``
3. **Route registration**: Mount health check and future API routes

The factory pattern allows dependency injection of config for testing.

Usage::

    from src.app import create_app
    from src.utils.config_loader import load_config

    config = load_config("config.yaml")
    app = create_app(config)

FastAPI 应用工厂和生命周期管理。

创建 FastAPI 应用实例，包含：

1. **生命周期管理**：启动时初始化 ServiceContext 和 ChatStorage，
   关闭时通过 ``close_all()`` 清理资源
2. **CORS 中间件**：从 ``server_config.cors`` 配置允许的来源
3. **路由注册**：挂载健康检查和未来的 API 路由

工厂模式允许为测试注入配置依赖。

Reference: docs/Phase5_执行规格.md §US-SRV-003,
docs/项目架构设计.md §2.5
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.service_context import ServiceContext
from src.storage.character_storage import CharacterStorage, get_default_character_avatar_dir
from src.storage.factory import create_chat_storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle: startup and shutdown.

    管理应用生命周期：启动和关闭。
    """
    # Startup: initialize shared resources
    # 启动：初始化共享资源
    config = app.state.config
    logger.info("Starting FastAPI application")

    # Create storage instance
    storage_config = config.get("storage", {})
    app.state.storage = create_chat_storage(storage_config)
    logger.info(f"Storage initialized: mode={storage_config.get('mode', 'json')}")

    # Create service context
    app.state.service_context = ServiceContext(config)
    logger.info("ServiceContext initialized")

    yield

    # Shutdown: cleanup resources
    # 关闭：清理资源
    logger.info("Shutting down FastAPI application")
    try:
        await app.state.service_context.close_all()
        logger.info("ServiceContext closed successfully")
    except Exception as e:
        logger.error(f"Error during ServiceContext shutdown: {e}")


def create_app(config: dict) -> FastAPI:
    """Create and configure FastAPI application.

    创建并配置 FastAPI 应用。

    Args:
        config: Application configuration dict (from config.yaml)
                应用配置字典（来自 config.yaml）

    Returns:
        Configured FastAPI instance
        已配置的 FastAPI 实例
    """
    app = FastAPI(
        title="Emotion Robot API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config in app state for lifespan access
    # 将配置存储在 app state 中供 lifespan 访问
    app.state.config = config
    app.state.character_storage = CharacterStorage()

    avatar_dir = get_default_character_avatar_dir()
    avatar_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/api/assets/avatars",
        StaticFiles(directory=str(avatar_dir), check_dir=False),
        name="character-avatar-assets",
    )

    # Configure CORS
    # 配置 CORS
    server_config = config.get("server", {})
    cors_config = server_config.get("cors", {})

    if cors_config.get("enabled", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("allow_origins", ["*"]),
            allow_credentials=cors_config.get("allow_credentials", True),
            allow_methods=cors_config.get("allow_methods", ["*"]),
            allow_headers=cors_config.get("allow_headers", ["*"]),
        )
        logger.info("CORS middleware enabled")

    # Register routes
    # 注册路由
    from src.routes.characters import router as characters_router
    from src.routes.chat_ws import websocket_endpoint
    from src.routes.chats import router as chats_router
    from src.routes.health import router as health_router

    app.include_router(health_router)
    app.include_router(characters_router)
    app.include_router(chats_router)

    # Register WebSocket endpoint
    # 注册 WebSocket 端点
    app.websocket("/ws")(websocket_endpoint)

    logger.info("FastAPI app created successfully")
    return app
