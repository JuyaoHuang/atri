"""
Character management REST API routes.
角色管理 REST API 路由。

Provides endpoints for listing and retrieving character personas.
提供角色列表和详情查询端点。
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..agent.persona import list_personas, load_persona

router = APIRouter(prefix="/api/characters", tags=["characters"])


class CharacterListItem(BaseModel):
    """Character list item (without system_prompt). / 角色列表项（不含系统提示词）"""

    character_id: str
    name: str
    avatar: str | None
    greeting: str | None


class CharacterDetail(BaseModel):
    """Character detail (with system_prompt). / 角色详情（含系统提示词）"""

    character_id: str
    name: str
    avatar: str | None
    greeting: str | None
    system_prompt: str


@router.get("", response_model=list[CharacterListItem])
async def list_characters() -> list[CharacterListItem]:
    """
    List all available characters (without system_prompt).
    列出所有可用角色（不含系统提示词）。

    Returns:
        List of character metadata (character_id, name, avatar, greeting).
        角色元数据列表（角色ID、名称、头像、问候语）。
    """
    character_ids = list_personas()
    characters = []

    for cid in character_ids:
        persona = load_persona(cid)
        characters.append(
            CharacterListItem(
                character_id=cid,
                name=persona.name,
                avatar=persona.avatar,
                greeting=persona.greeting,
            )
        )

    return characters


@router.get("/{character_id}", response_model=CharacterDetail)
async def get_character(character_id: str) -> CharacterDetail:
    """
    Get character detail (with system_prompt).
    获取角色详情（含系统提示词）。

    Args:
        character_id: Character identifier.
                      角色标识符。

    Returns:
        Full character metadata including system_prompt.
        完整角色元数据（含系统提示词）。

    Raises:
        HTTPException: 404 if character not found.
                       角色不存在时返回 404。
    """
    try:
        persona = load_persona(character_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found") from e

    return CharacterDetail(
        character_id=character_id,
        name=persona.name,
        avatar=persona.avatar,
        greeting=persona.greeting,
        system_prompt=persona.system_prompt,
    )
