"""Live integration tests for FastAPI server.
FastAPI 服务器的 live 集成测试。

These tests make real LLM calls and WebSocket connections.
Run with: uv run pytest tests/routes/test_live_server.py -m live -v -s

这些测试进行真实的 LLM 调用和 WebSocket 连接。
运行命令：uv run pytest tests/routes/test_live_server.py -m live -v -s

Reference: docs/Phase5_执行规格.md §US-SRV-008
"""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from src.app import create_app
from src.utils.config_loader import load_config


@pytest.fixture
def live_app():
    """Create FastAPI app with real config for live testing.
    使用真实配置创建 FastAPI 应用用于 live 测试。
    """
    config = load_config("config.yaml")
    app = create_app(config)
    return app


@pytest.mark.live
@pytest.mark.asyncio
async def test_health_endpoint(live_app):
    """Test health check endpoint.
    测试健康检查端点。
    """
    async with AsyncClient(transport=ASGITransport(app=live_app), base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.live
@pytest.mark.asyncio
async def test_characters_endpoint(live_app):
    """Test characters list endpoint.
    测试角色列表端点。
    """
    async with AsyncClient(transport=ASGITransport(app=live_app), base_url="http://test") as client:
        response = await client.get("/api/characters")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should contain at least atri character
        # 应该至少包含 atri 角色
        character_ids = [char["character_id"] for char in data]
        assert "atri" in character_ids


@pytest.mark.live
@pytest.mark.asyncio
async def test_create_chat_with_llm_title(live_app):
    """Test chat creation with LLM-generated title.
    测试使用 LLM 生成标题创建聊天。
    """
    async with AsyncClient(transport=ASGITransport(app=live_app), base_url="http://test") as client:
        # Manually initialize app.state (ASGITransport doesn't trigger lifespan)
        # 手动初始化 app.state（ASGITransport 不触发 lifespan）
        from src.service_context import ServiceContext
        from src.storage.factory import create_chat_storage

        config = live_app.state.config
        live_app.state.storage = create_chat_storage(config.get("storage", {}))
        live_app.state.service_context = ServiceContext(config)

        # Create chat
        # 创建聊天
        response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "你好，今天天气真好"},
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "title" in data
        assert data["title"]  # Title should not be empty
        assert data["character_id"] == "atri"

        chat_id = data["id"]

        # Cleanup: delete the chat
        # 清理：删除聊天
        await client.post(f"/api/chats/{chat_id}/delete")


@pytest.mark.live
def test_websocket_three_rounds_conversation(live_app):
    """Test WebSocket connection with 3 rounds of conversation.
    测试 WebSocket 连接进行 3 轮对话。
    """
    # Manually initialize app.state (TestClient doesn't trigger lifespan)
    # 手动初始化 app.state（TestClient 不触发 lifespan）
    from src.service_context import ServiceContext
    from src.storage.factory import create_chat_storage

    config = live_app.state.config
    live_app.state.storage = create_chat_storage(config.get("storage", {}))
    live_app.state.service_context = ServiceContext(config)

    client = TestClient(live_app)

    # Create a chat first via REST API
    # 先通过 REST API 创建聊天
    response = client.post(
        "/api/chats",
        json={"character_id": "atri", "first_message": "测试"},
    )
    assert response.status_code == 201
    chat_id = response.json()["id"]

    try:
        with client.websocket_connect("/ws") as websocket:
            # Round 1
            # 第 1 轮
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {
                        "text": "你好",
                        "chat_id": chat_id,
                        "character_id": "atri",
                    },
                }
            )

            # Receive chunks until complete
            # 接收 chunks 直到完成
            chunks_received = 0
            while True:
                response = websocket.receive_json()
                if response["type"] == "output:chat:chunk":
                    chunks_received += 1
                elif response["type"] == "output:chat:complete":
                    assert "full_reply" in response["data"]
                    break

            assert chunks_received > 0

            # Round 2
            # 第 2 轮
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {
                        "text": "今天天气怎么样",
                        "chat_id": chat_id,
                        "character_id": "atri",
                    },
                }
            )

            while True:
                response = websocket.receive_json()
                if response["type"] == "output:chat:complete":
                    break

            # Round 3
            # 第 3 轮
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {
                        "text": "谢谢",
                        "chat_id": chat_id,
                        "character_id": "atri",
                    },
                }
            )

            while True:
                response = websocket.receive_json()
                if response["type"] == "output:chat:complete":
                    break

        # Verify message history contains 6 messages (3 human + 3 ai)
        # 验证消息历史包含 6 条消息（3 human + 3 ai）
        response = client.get(f"/api/chats/{chat_id}")
        assert response.status_code == 200
        data = response.json()
        messages = data["messages"]
        assert len(messages) >= 6  # At least 6 messages

        # Count human and ai messages
        # 统计 human 和 ai 消息
        human_count = sum(1 for msg in messages if msg["role"] == "human")
        ai_count = sum(1 for msg in messages if msg["role"] == "ai")
        assert human_count >= 3
        assert ai_count >= 3

    finally:
        # Cleanup: delete the chat
        # 清理：删除聊天
        client.post(f"/api/chats/{chat_id}/delete")


@pytest.mark.live
@pytest.mark.asyncio
async def test_chat_update_and_delete(live_app):
    """Test chat title update and deletion.
    测试聊天标题更新和删除。
    """
    async with AsyncClient(transport=ASGITransport(app=live_app), base_url="http://test") as client:
        # Manually initialize app.state
        # 手动初始化 app.state
        from src.service_context import ServiceContext
        from src.storage.factory import create_chat_storage

        config = live_app.state.config
        live_app.state.storage = create_chat_storage(config.get("storage", {}))
        live_app.state.service_context = ServiceContext(config)

        # Create chat
        # 创建聊天
        response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "测试更新和删除"},
        )
        assert response.status_code == 201
        chat_id = response.json()["id"]

        # Update title
        # 更新标题
        response = await client.post(
            f"/api/chats/{chat_id}/update",
            json={"title": "新标题测试"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "新标题测试"

        # Delete chat
        # 删除聊天
        response = await client.post(f"/api/chats/{chat_id}/delete")
        assert response.status_code == 204

        # Verify chat is deleted
        # 验证聊天已删除
        response = await client.get(f"/api/chats/{chat_id}")
        assert response.status_code == 404
