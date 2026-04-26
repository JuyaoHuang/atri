"""Tests for chat session CRUD REST API.
聊天会话 CRUD REST API 测试。
"""

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.service_context import ServiceContext
from src.storage.factory import create_chat_storage
from src.utils.config_loader import load_config


@pytest_asyncio.fixture
async def client():
    """Create test client with manually initialized storage.
    创建测试客户端，手动初始化 storage。
    """
    config = load_config("config.yaml")
    app = create_app(config)

    # Manually initialize storage and service_context
    # (since lifespan is not triggered by ASGITransport)
    # 手动初始化 storage 和 service_context
    # （因为 ASGITransport 不会触发 lifespan）
    storage_config = config.get("storage", {})
    app.state.storage = create_chat_storage(storage_config)
    app.state.service_context = ServiceContext(config)

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as ac:
        yield ac

    # Cleanup
    # 清理
    await app.state.service_context.close_all()


@pytest.mark.asyncio
async def test_create_chat_with_llm_title(client: AsyncClient):
    """Create chat should generate title with LLM.
    创建聊天应使用 LLM 生成标题。
    """

    # Mock LLM title generation
    # 模拟 LLM 标题生成
    async def mock_stream(*args, **kwargs):
        yield "与"
        yield "亚托莉"
        yield "的"
        yield "问候"

    with patch("src.routes.chats.create_from_role") as mock_create:
        mock_llm = MagicMock()
        # Use side_effect with a callable that returns the async generator
        # 使用 side_effect 配合返回 async generator 的可调用对象
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "你好，亚托莉！"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["character_id"] == "atri"
    assert data["title"] == "与亚托莉的问候"
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


@pytest.mark.asyncio
async def test_create_chat_fallback_on_llm_failure(client: AsyncClient):
    """Create chat should use fallback title when LLM fails.
    LLM 失败时创建聊天应使用降级标题。
    """
    # Mock LLM to raise error
    # 模拟 LLM 抛出错误
    with patch("src.routes.chats.create_from_role") as mock_create:
        from src.llm.exceptions import LLMConnectionError

        mock_create.side_effect = LLMConnectionError("Connection failed")

        response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "你好，亚托莉！今天天气真好"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["character_id"] == "atri"
    # Should use first 20 chars as fallback
    # 应使用前 20 字符作为降级标题
    assert data["title"] == "你好，亚托莉！今天天气真好"
    assert len(data["title"]) <= 20


@pytest.mark.asyncio
async def test_create_chat_truncates_long_fallback(client: AsyncClient):
    """Fallback title should truncate long messages.
    降级标题应截断长消息。
    """
    with patch("src.routes.chats.create_from_role") as mock_create:
        from src.llm.exceptions import LLMAPIError

        mock_create.side_effect = LLMAPIError("API error")

        long_message = "这是一条非常非常非常非常非常非常长的消息，超过了20个字符的限制"
        response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": long_message},
        )

    assert response.status_code == 201
    data = response.json()
    assert len(data["title"]) == 20
    assert data["title"] == long_message[:20]


@pytest.mark.asyncio
async def test_create_chat_with_deferred_title_uses_first_sentence_excerpt(client: AsyncClient):
    """Deferred title mode should return the first sentence's first 8 chars."""

    async def mock_stream(*args, **kwargs):
        yield "真正标题"

    with patch("src.routes.chats.create_from_role") as mock_create:
        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        response = await client.post(
            "/api/chats",
            json={
                "character_id": "atri",
                "first_message": "你好呀亚托莉。今天一起散步吧！",
                "defer_title": True,
            },
        )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "你好呀亚托莉"


@pytest.mark.asyncio
async def test_create_chat_with_deferred_title_backfills_generated_title(client: AsyncClient):
    """Deferred title mode should patch the stored title in the background."""

    async def mock_stream(*args, **kwargs):
        yield "延后生成标题"

    with patch("src.routes.chats.create_from_role") as mock_create:
        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        create_response = await client.post(
            "/api/chats",
            json={
                "character_id": "atri",
                "first_message": "晚上一起看海吧。顺便聊聊心情。",
                "defer_title": True,
            },
        )

    assert create_response.status_code == 201
    create_data = create_response.json()
    assert create_data["title"] == "晚上一起看海吧"

    detail_response = await client.get(f"/api/chats/{create_data['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["metadata"]["title"] == "延后生成标题"


@pytest.mark.asyncio
async def test_list_chats_empty(client: AsyncClient):
    """List chats should return empty list for new user.
    新用户列出聊天应返回空列表。
    """
    response = await client.get("/api/chats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # May have chats from previous tests, just check it's a list
    # 可能有之前测试的聊天，只检查是否为列表


@pytest.mark.asyncio
async def test_list_chats_with_character_filter(client: AsyncClient):
    """List chats should support character_id filter.
    列出聊天应支持 character_id 过滤。
    """
    # Create a chat first
    # 先创建一个聊天
    with patch("src.routes.chats.create_from_role") as mock_create:

        async def mock_stream(*args, **kwargs):
            yield "测试标题"

        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "测试消息"},
        )

    # List with filter
    # 使用过滤器列出
    response = await client.get("/api/chats?character_id=atri")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # All returned chats should be for atri
    # 所有返回的聊天都应该是 atri 的
    for chat in data:
        assert chat["character_id"] == "atri"


@pytest.mark.asyncio
async def test_list_chats_sorted_by_updated_at(client: AsyncClient):
    """List chats should be sorted by updated_at descending.
    列出聊天应按 updated_at 降序排序。
    """
    with patch("src.routes.chats.create_from_role") as mock_create:

        async def mock_stream(*args, **kwargs):
            yield "标题"

        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        # Create two chats
        # 创建两个聊天
        await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "第一条"},
        )
        await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "第二条"},
        )

    response = await client.get("/api/chats?character_id=atri")
    assert response.status_code == 200
    data = response.json()

    if len(data) >= 2:
        # Check descending order
        # 检查降序
        for i in range(len(data) - 1):
            assert data[i]["updated_at"] >= data[i + 1]["updated_at"]


@pytest.mark.asyncio
async def test_get_chat_details(client: AsyncClient):
    """Get chat should return metadata and messages.
    获取聊天应返回元数据和消息。
    """
    # Create a chat
    # 创建一个聊天
    with patch("src.routes.chats.create_from_role") as mock_create:

        async def mock_stream(*args, **kwargs):
            yield "测试聊天"

        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        create_response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "你好"},
        )
        chat_id = create_response.json()["id"]

    # Get chat details
    # 获取聊天详情
    response = await client.get(f"/api/chats/{chat_id}")
    assert response.status_code == 200
    data = response.json()

    assert "metadata" in data
    assert "messages" in data
    assert data["metadata"]["id"] == chat_id
    assert data["metadata"]["character_id"] == "atri"
    assert isinstance(data["messages"], list)


@pytest.mark.asyncio
async def test_get_chat_not_found(client: AsyncClient):
    """Get chat should return 404 for nonexistent chat.
    获取不存在的聊天应返回 404。
    """
    response = await client.get("/api/chats/nonexistent_chat_id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_chat_with_pagination(client: AsyncClient):
    """Get chat should support limit and offset pagination.
    获取聊天应支持 limit 和 offset 分页。
    """
    # Create a chat
    # 创建一个聊天
    with patch("src.routes.chats.create_from_role") as mock_create:

        async def mock_stream(*args, **kwargs):
            yield "分页测试"

        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        create_response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "分页测试"},
        )
        chat_id = create_response.json()["id"]

    # Test pagination parameters
    # 测试分页参数
    response = await client.get(f"/api/chats/{chat_id}?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) <= 10


@pytest.mark.asyncio
async def test_update_chat_title(client: AsyncClient):
    """Update chat title should work.
    更新聊天标题应该有效。
    """
    # Create a chat
    # 创建一个聊天
    with patch("src.routes.chats.create_from_role") as mock_create:

        async def mock_stream(*args, **kwargs):
            yield "原标题"

        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        create_response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "测试"},
        )
        chat_id = create_response.json()["id"]

    # Update title
    # 更新标题
    response = await client.post(
        f"/api/chats/{chat_id}/update",
        json={"title": "新标题"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "新标题"
    assert data["id"] == chat_id


@pytest.mark.asyncio
async def test_update_chat_not_found(client: AsyncClient):
    """Update chat should return 404 for nonexistent chat.
    更新不存在的聊天应返回 404。
    """
    response = await client.post(
        "/api/chats/nonexistent_chat_id/update",
        json={"title": "新标题"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_chat(client: AsyncClient):
    """Delete chat should work and return 204.
    删除聊天应该有效并返回 204。
    """
    # Create a chat
    # 创建一个聊天
    with patch("src.routes.chats.create_from_role") as mock_create:

        async def mock_stream(*args, **kwargs):
            yield "待删除"

        mock_llm = MagicMock()
        mock_llm.chat_completion_stream = MagicMock(
            side_effect=lambda *a, **kw: mock_stream(*a, **kw)
        )
        mock_create.return_value = mock_llm

        create_response = await client.post(
            "/api/chats",
            json={"character_id": "atri", "first_message": "待删除"},
        )
        chat_id = create_response.json()["id"]

    # Delete chat
    # 删除聊天
    response = await client.post(f"/api/chats/{chat_id}/delete")
    assert response.status_code == 204

    # Verify chat is deleted
    # 验证聊天已删除
    get_response = await client.get(f"/api/chats/{chat_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_chat_not_found(client: AsyncClient):
    """Delete chat should return 404 for nonexistent chat.
    删除不存在的聊天应返回 404。
    """
    response = await client.post("/api/chats/nonexistent_chat_id/delete")
    assert response.status_code == 404
