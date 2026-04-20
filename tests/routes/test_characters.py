"""
Tests for character management REST API.
角色管理 REST API 测试。
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.utils.config_loader import load_config


@pytest_asyncio.fixture
async def client():
    """Create test client. / 创建测试客户端。"""
    config = load_config("config.yaml")
    app = create_app(config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_list_characters_returns_at_least_one(client: AsyncClient):
    """List should return at least 1 character (atri). / 列表应至少返回 1 个角色（atri）。"""
    response = await client.get("/api/characters")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    # Check atri exists
    atri = next((c for c in data if c["character_id"] == "atri"), None)
    assert atri is not None
    assert "name" in atri
    assert "avatar" in atri
    assert "greeting" in atri


@pytest.mark.asyncio
async def test_list_characters_excludes_system_prompt(client: AsyncClient):
    """List should not include system_prompt. / 列表不应包含 system_prompt。"""
    response = await client.get("/api/characters")
    assert response.status_code == 200

    data = response.json()
    for character in data:
        assert "system_prompt" not in character


@pytest.mark.asyncio
async def test_get_character_returns_full_fields(client: AsyncClient):
    """Detail should return all fields including system_prompt.

    详情应返回所有字段（含 system_prompt）。
    """
    response = await client.get("/api/characters/atri")
    assert response.status_code == 200

    data = response.json()
    assert data["character_id"] == "atri"
    assert "name" in data
    assert "avatar" in data
    assert "greeting" in data
    assert "system_prompt" in data
    assert isinstance(data["system_prompt"], str)
    assert len(data["system_prompt"]) > 0


@pytest.mark.asyncio
async def test_get_nonexistent_character_returns_404(client: AsyncClient):
    """Nonexistent character should return 404. / 不存在的角色应返回 404。"""
    response = await client.get("/api/characters/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
