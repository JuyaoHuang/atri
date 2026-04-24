"""Tests for Phase 7 character management routes."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.storage.character_storage import CharacterStorage
from src.utils.config_loader import load_config


def _write_persona(
    path: Path,
    *,
    name: str,
    avatar: str | None,
    greeting: str | None,
    description: str | None,
    system_prompt: str,
    created_at: str | None = None,
    updated_at: str | None = None,
    managed_by: str | None = None,
) -> None:
    metadata: dict[str, str] = {"name": name}
    if avatar is not None:
        metadata["avatar"] = avatar
    if greeting is not None:
        metadata["greeting"] = greeting
    if description is not None:
        metadata["description"] = description
    if created_at is not None:
        metadata["created_at"] = created_at
    if updated_at is not None:
        metadata["updated_at"] = updated_at
    if managed_by is not None:
        metadata["managed_by"] = managed_by

    lines = ["---"]
    lines.extend(f"{key}: {value}" for key, value in metadata.items())
    lines.extend(["---", "", system_prompt, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest_asyncio.fixture
async def client_and_storage(tmp_path: Path):
    """Create test client with isolated character storage."""

    persona_dir = tmp_path / "persona"
    avatar_dir = tmp_path / "avatars"
    persona_dir.mkdir(parents=True, exist_ok=True)

    _write_persona(
        persona_dir / "atri.md",
        name="亚托莉",
        avatar="atri.jpg",
        greeting="主人，早上好！",
        description="默认角色",
        system_prompt="你是亚托莉。",
    )
    _write_persona(
        persona_dir / "bilibili.md",
        name="御坂美琴",
        avatar="bilibili.jpg",
        greeting="今天也要加油。",
        description="备用角色",
        system_prompt="你是御坂美琴。",
    )

    config = load_config("config.yaml")
    app = create_app(config)
    storage = CharacterStorage(persona_dir=persona_dir, avatar_dir=avatar_dir)
    app.state.character_storage = storage

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, storage


@pytest.mark.asyncio
async def test_list_characters_returns_existing_personas(client_and_storage):
    client, _storage = client_and_storage

    response = await client.get("/api/characters")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert {item["character_id"] for item in data} == {"atri", "bilibili"}
    assert all("system_prompt" not in item for item in data)
    assert all(item["is_system"] is True for item in data)


@pytest.mark.asyncio
async def test_get_character_returns_full_fields(client_and_storage):
    client, _storage = client_and_storage

    response = await client.get("/api/characters/atri")
    assert response.status_code == 200

    data = response.json()
    assert data["character_id"] == "atri"
    assert data["name"] == "亚托莉"
    assert data["avatar"] == "atri.jpg"
    assert data["description"] == "默认角色"
    assert data["system_prompt"] == "你是亚托莉。"
    assert data["avatar_url"] is None
    assert data["is_system"] is True


@pytest.mark.asyncio
async def test_create_character_persists_markdown_file(client_and_storage):
    client, storage = client_and_storage

    response = await client.post(
        "/api/characters",
        json={
            "name": "Phase7 Card",
            "greeting": "你好，我是新角色。",
            "description": "Phase 7 创建的角色",
            "system_prompt": "你是 Phase 7 新角色。",
        },
    )
    assert response.status_code == 201

    data = response.json()
    assert data["character_id"] == "phase7-card"
    assert data["is_system"] is False
    assert data["description"] == "Phase 7 创建的角色"
    assert data["created_at"] is not None
    assert storage.get_character("phase7-card").managed_by == "atri"
    assert (storage.persona_dir / "phase7-card.md").is_file()


@pytest.mark.asyncio
async def test_update_character_changes_metadata(client_and_storage):
    client, _storage = client_and_storage

    create_response = await client.post(
        "/api/characters",
        json={
            "name": "Editable Card",
            "greeting": "初始问候",
            "description": "初始描述",
            "system_prompt": "初始设定",
        },
    )
    created = create_response.json()

    update_response = await client.put(
        f"/api/characters/{created['character_id']}",
        json={
            "name": "Editable Card Updated",
            "greeting": "更新后的问候",
            "description": "更新后的描述",
            "system_prompt": "更新后的设定",
        },
    )
    assert update_response.status_code == 200

    updated = update_response.json()
    assert updated["name"] == "Editable Card Updated"
    assert updated["greeting"] == "更新后的问候"
    assert updated["description"] == "更新后的描述"
    assert updated["system_prompt"] == "更新后的设定"
    assert updated["updated_at"] is not None


@pytest.mark.asyncio
async def test_delete_character_removes_custom_persona_file(client_and_storage):
    client, storage = client_and_storage

    create_response = await client.post(
        "/api/characters",
        json={
            "name": "Disposable Card",
            "greeting": "待删除",
            "description": "待删除角色",
            "system_prompt": "删除测试",
        },
    )
    created = create_response.json()

    delete_response = await client.delete(f"/api/characters/{created['character_id']}")
    assert delete_response.status_code == 204
    assert not (storage.persona_dir / f"{created['character_id']}.md").exists()

    get_response = await client.get(f"/api/characters/{created['character_id']}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_system_character_returns_400(client_and_storage):
    client, _storage = client_and_storage

    response = await client.delete("/api/characters/atri")
    assert response.status_code == 400
    assert "protected system character" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_character_rejects_duplicate_name(client_and_storage):
    client, _storage = client_and_storage

    response = await client.post(
        "/api/characters",
        json={
            "name": "亚托莉",
            "greeting": "重复",
            "description": "重复名称",
            "system_prompt": "重复名称测试",
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_avatar_stores_file_and_updates_character(client_and_storage):
    client, storage = client_and_storage

    create_response = await client.post(
        "/api/characters",
        json={
            "name": "Avatar Card",
            "greeting": "头像测试",
            "description": "测试头像上传",
            "system_prompt": "头像测试角色",
        },
    )
    created = create_response.json()

    upload_response = await client.post(
        f"/api/characters/{created['character_id']}/avatar",
        files={"avatar": ("avatar.png", b"fake-image-content", "image/png")},
    )
    assert upload_response.status_code == 200

    payload = upload_response.json()
    assert payload["character_id"] == created["character_id"]
    assert payload["avatar"].endswith(".png")
    assert payload["avatar_url"].startswith("http://test/api/assets/avatars/")
    assert (storage.avatar_dir / payload["avatar"]).is_file()

    detail_response = await client.get(f"/api/characters/{created['character_id']}")
    detail = detail_response.json()
    assert detail["avatar"] == payload["avatar"]
    assert detail["avatar_url"] == payload["avatar_url"]
