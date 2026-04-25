"""Tests for WebSocket chat endpoint.
WebSocket 聊天端点测试。

Reference: docs/Phase5_执行规格.md §US-SRV-006
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.app import create_app


@pytest.fixture
def mock_config() -> dict:
    """Mock configuration for testing.
    测试用的模拟配置。
    """
    return {
        "server": {
            "cors": {
                "enabled": True,
                "allow_origins": ["*"],
                "allow_methods": ["*"],
                "allow_credentials": True,
            }
        },
        "storage": {"mode": "json", "json": {"base_path": "data/chats"}},
        "llm": {},
        "memory": {},
    }


@pytest.fixture
def mock_service_context():
    """Mock ServiceContext with ChatAgent.
    模拟 ServiceContext 和 ChatAgent。
    """
    mock_agent = MagicMock()
    mock_context = MagicMock()
    mock_context.get_or_create_agent.return_value = mock_agent
    return mock_context, mock_agent


@pytest.fixture
def mock_storage():
    """Mock ChatStorage.
    模拟 ChatStorage。
    """
    storage = AsyncMock()
    storage.append_message = AsyncMock()
    return storage


async def _mock_chat_stream(chunks: list[str]) -> AsyncIterator[str]:
    """Helper to create async generator for mocking ChatAgent.chat().
    创建异步生成器用于模拟 ChatAgent.chat()。
    """
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_websocket_connection_and_ping_pong(
    mock_config: dict, mock_service_context: tuple, mock_storage: AsyncMock
) -> None:
    """Test WebSocket connection establishment and ping/pong heartbeat.
    测试 WebSocket 连接建立和 ping/pong 心跳。
    """
    mock_context, _ = mock_service_context

    with (
        patch("src.app.ServiceContext", return_value=mock_context),
        patch("src.app.create_chat_storage", return_value=mock_storage),
    ):
        app = create_app(mock_config)

        # Manually set app.state since TestClient doesn't trigger lifespan
        # 手动设置 app.state，因为 TestClient 不触发 lifespan
        app.state.service_context = mock_context
        app.state.storage = mock_storage

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send ping
            # 发送 ping
            websocket.send_json({"type": "ping"})

            # Receive pong
            # 接收 pong
            response = websocket.receive_json()
            assert response == {"type": "pong"}


@pytest.mark.asyncio
async def test_websocket_text_input_streaming(
    mock_config: dict, mock_service_context: tuple, mock_storage: AsyncMock
) -> None:
    """Test WebSocket text input with streaming response.
    测试 WebSocket 文本输入和流式响应。
    """
    mock_context, mock_agent = mock_service_context

    # Mock ChatAgent.chat() to return streaming chunks
    # 模拟 ChatAgent.chat() 返回流式 chunks
    chunks = ["你好", "，", "主人", "！"]
    mock_agent.chat = MagicMock(side_effect=lambda text: _mock_chat_stream(chunks))

    with (
        patch("src.app.ServiceContext", return_value=mock_context),
        patch("src.app.create_chat_storage", return_value=mock_storage),
    ):
        app = create_app(mock_config)

        # Manually set app.state since TestClient doesn't trigger lifespan
        # 手动设置 app.state，因为 TestClient 不触发 lifespan
        app.state.service_context = mock_context
        app.state.storage = mock_storage

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send text input
            # 发送文本输入
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {
                        "text": "你好",
                        "chat_id": "test_chat_123",
                        "character_id": "atri",
                    },
                }
            )

            # Receive streaming chunks
            # 接收流式 chunks
            received_chunks = []
            for _ in range(len(chunks)):
                response = websocket.receive_json()
                assert response["type"] == "output:chat:chunk"
                assert response["data"]["chat_id"] == "test_chat_123"
                assert response["data"]["character_id"] == "atri"
                received_chunks.append(response["data"]["chunk"])

            # Receive complete message
            # 接收完成消息
            complete_response = websocket.receive_json()
            assert complete_response["type"] == "output:chat:complete"
            assert complete_response["data"]["full_reply"] == "".join(chunks)
            assert complete_response["data"]["chat_id"] == "test_chat_123"
            assert complete_response["data"]["character_id"] == "atri"

            # Verify ChatAgent was called
            # 验证 ChatAgent 被调用
            mock_context.get_or_create_agent.assert_called_once_with("atri", "default")
            mock_agent.chat.assert_called_once_with("你好")

            # Verify messages were persisted
            # 验证消息被持久化
            assert mock_storage.append_message.call_count == 2
            mock_storage.append_message.assert_any_call(
                "test_chat_123", "human", "你好", name="default"
            )
            mock_storage.append_message.assert_any_call(
                "test_chat_123", "ai", "你好，主人！", name="atri"
            )


@pytest.mark.asyncio
async def test_websocket_text_input_passes_client_context_without_persisting_it(
    mock_config: dict, mock_service_context: tuple, mock_storage: AsyncMock
) -> None:
    mock_context, mock_agent = mock_service_context
    chunks = ["It is noon."]
    client_context = {
        "datetime": {
            "iso": "2026-04-25T04:00:00.000Z",
            "local": "2026/4/25 12:00:00",
            "time_zone": "Asia/Shanghai",
            "utc_offset": "UTC+08:00",
        }
    }
    mock_agent.chat = MagicMock(
        side_effect=lambda text, runtime_context=None: _mock_chat_stream(chunks)
    )

    with (
        patch("src.app.ServiceContext", return_value=mock_context),
        patch("src.app.create_chat_storage", return_value=mock_storage),
    ):
        app = create_app(mock_config)
        app.state.service_context = mock_context
        app.state.storage = mock_storage

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {
                        "text": "what time is it?",
                        "chat_id": "test_chat_time",
                        "character_id": "atri",
                        "client_context": client_context,
                    },
                }
            )

            chunk_response = websocket.receive_json()
            assert chunk_response["type"] == "output:chat:chunk"
            complete_response = websocket.receive_json()
            assert complete_response["type"] == "output:chat:complete"

            mock_agent.chat.assert_called_once_with(
                "what time is it?",
                runtime_context=client_context,
            )
            mock_storage.append_message.assert_any_call(
                "test_chat_time", "human", "what time is it?", name="default"
            )
            persisted_user_args = mock_storage.append_message.call_args_list[0].args
            assert persisted_user_args == ("test_chat_time", "human", "what time is it?")


@pytest.mark.asyncio
async def test_websocket_invalid_json(
    mock_config: dict, mock_service_context: tuple, mock_storage: AsyncMock
) -> None:
    """Test WebSocket handling of invalid JSON (should send error but not disconnect).
    测试 WebSocket 处理无效 JSON（应发送错误但不断开连接）。
    """
    mock_context, _ = mock_service_context

    with (
        patch("src.app.ServiceContext", return_value=mock_context),
        patch("src.app.create_chat_storage", return_value=mock_storage),
    ):
        app = create_app(mock_config)

        # Manually set app.state since TestClient doesn't trigger lifespan
        # 手动设置 app.state，因为 TestClient 不触发 lifespan
        app.state.service_context = mock_context
        app.state.storage = mock_storage

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send invalid JSON
            # 发送无效 JSON
            websocket.send_text("{invalid json")

            # Receive error message
            # 接收错误消息
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid JSON" in response["data"]["message"]

            # Connection should still be alive - send ping to verify
            # 连接应该仍然存活 - 发送 ping 验证
            websocket.send_json({"type": "ping"})
            pong_response = websocket.receive_json()
            assert pong_response == {"type": "pong"}


@pytest.mark.asyncio
async def test_websocket_unknown_message_type(
    mock_config: dict, mock_service_context: tuple, mock_storage: AsyncMock
) -> None:
    """Test WebSocket handling of unknown message type (should send error but not disconnect).
    测试 WebSocket 处理未知消息类型（应发送错误但不断开连接）。
    """
    mock_context, _ = mock_service_context

    with (
        patch("src.app.ServiceContext", return_value=mock_context),
        patch("src.app.create_chat_storage", return_value=mock_storage),
    ):
        app = create_app(mock_config)

        # Manually set app.state since TestClient doesn't trigger lifespan
        # 手动设置 app.state，因为 TestClient 不触发 lifespan
        app.state.service_context = mock_context
        app.state.storage = mock_storage

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send unknown message type
            # 发送未知消息类型
            websocket.send_json({"type": "unknown:type", "data": {}})

            # Receive error message
            # 接收错误消息
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Unknown message type" in response["data"]["message"]

            # Connection should still be alive
            # 连接应该仍然存活
            websocket.send_json({"type": "ping"})
            pong_response = websocket.receive_json()
            assert pong_response == {"type": "pong"}


@pytest.mark.asyncio
async def test_websocket_missing_required_fields(
    mock_config: dict, mock_service_context: tuple, mock_storage: AsyncMock
) -> None:
    """Test WebSocket handling of missing required fields in input:text.
    测试 WebSocket 处理 input:text 中缺失必填字段。
    """
    mock_context, _ = mock_service_context

    with (
        patch("src.app.ServiceContext", return_value=mock_context),
        patch("src.app.create_chat_storage", return_value=mock_storage),
    ):
        app = create_app(mock_config)

        # Manually set app.state since TestClient doesn't trigger lifespan
        # 手动设置 app.state，因为 TestClient 不触发 lifespan
        app.state.service_context = mock_context
        app.state.storage = mock_storage

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Missing 'text' field
            # 缺失 'text' 字段
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {"chat_id": "test_123", "character_id": "atri"},
                }
            )
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Missing 'text'" in response["data"]["message"]

            # Missing 'chat_id' field
            # 缺失 'chat_id' 字段
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {"text": "hello", "character_id": "atri"},
                }
            )
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Missing 'chat_id'" in response["data"]["message"]

            # Missing 'character_id' field
            # 缺失 'character_id' 字段
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {"text": "hello", "chat_id": "test_123"},
                }
            )
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Missing 'character_id'" in response["data"]["message"]


@pytest.mark.asyncio
async def test_websocket_storage_persistence_failure(
    mock_config: dict, mock_service_context: tuple, mock_storage: AsyncMock
) -> None:
    """Test WebSocket handling when storage persistence fails (should not crash).
    测试存储持久化失败时的 WebSocket 处理（不应崩溃）。
    """
    mock_context, mock_agent = mock_service_context

    # Mock ChatAgent.chat() to return chunks
    # 模拟 ChatAgent.chat() 返回 chunks
    chunks = ["Hello"]
    mock_agent.chat = MagicMock(side_effect=lambda text: _mock_chat_stream(chunks))

    # Mock storage to raise ValueError (chat not found)
    # 模拟 storage 抛出 ValueError（聊天不存在）
    mock_storage.append_message.side_effect = ValueError("Chat not found")

    with (
        patch("src.app.ServiceContext", return_value=mock_context),
        patch("src.app.create_chat_storage", return_value=mock_storage),
    ):
        app = create_app(mock_config)

        # Manually set app.state since TestClient doesn't trigger lifespan
        # 手动设置 app.state，因为 TestClient 不触发 lifespan
        app.state.service_context = mock_context
        app.state.storage = mock_storage

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send text input
            # 发送文本输入
            websocket.send_json(
                {
                    "type": "input:text",
                    "data": {
                        "text": "test",
                        "chat_id": "nonexistent_chat",
                        "character_id": "atri",
                    },
                }
            )

            # Should still receive streaming response
            # 应该仍然收到流式响应
            chunk_response = websocket.receive_json()
            assert chunk_response["type"] == "output:chat:chunk"

            complete_response = websocket.receive_json()
            assert complete_response["type"] == "output:chat:complete"

            # Connection should still be alive (storage failure is logged but not fatal)
            # 连接应该仍然存活（存储失败被记录但不致命）
            websocket.send_json({"type": "ping"})
            pong_response = websocket.receive_json()
            assert pong_response == {"type": "pong"}
