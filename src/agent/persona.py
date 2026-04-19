"""Persona dataclass + loader for character-level configuration.

Persona files live at ``prompts/persona/{character_id}.md`` and use Markdown
with an optional YAML frontmatter header::

    ---
    name: 亚托莉
    avatar: atri.png
    greeting: 主人，早上好！
    ---

    # 角色设定
    你是亚托莉...

Frontmatter carries short display metadata (``name`` / ``avatar`` /
``greeting``); the body after the closing ``---`` is used verbatim as the LLM
system prompt. ``character_id`` is always taken from the function argument
(not from frontmatter), so renaming files does not break references.

Reference: docs/Phase4_执行规格.md §US-AGT-001 (S2 decision: Markdown +
frontmatter).

Persona dataclass + 角色配置加载器。

Persona 文件存于 ``prompts/persona/{character_id}.md``，采用带可选 YAML
frontmatter 头部的 Markdown 格式（示例同上）。

Frontmatter 承载简短的展示型元数据（``name`` / ``avatar`` / ``greeting``）；
closing ``---`` 之后的正文原样作为 LLM system prompt。``character_id`` 始终
从函数参数取（不从 frontmatter 读），这样重命名文件不会破坏引用。

参考：docs/Phase4_执行规格.md §US-AGT-001（S2 决策：Markdown + frontmatter）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from prompts.prompt_loader import load_persona as _load_persona_text


@dataclass(frozen=True)
class Persona:
    """Character-level configuration consumed by :class:`ChatAgent`.

    Attributes:
        character_id: Stable identifier used across chat_history, memory
            store paths, and mem0 ``agent_id``. Always equals the argument
            passed to :func:`load_persona`.
        name: Display name used as the chat_history ``name`` field for AI
            replies. Defaults to ``character_id`` when frontmatter omits it.
        avatar: Optional avatar filename for frontend rendering.
        greeting: Optional first-meeting greeting for frontend first screen.
        system_prompt: Markdown body injected as LLM payload position [1]
            (via :meth:`MemoryManager.build_llm_context`).

    ChatAgent 消费的角色级配置。

    属性：
        character_id：在 chat_history、记忆存储路径、mem0 ``agent_id`` 中使用
            的稳定标识。始终等于传给 :func:`load_persona` 的参数。
        name：chat_history 中 AI 回复的 ``name`` 字段（展示名）。当
            frontmatter 未提供时回退为 ``character_id``。
        avatar：可选的头像文件名，供前端渲染。
        greeting：可选的首次问候语，供前端首屏使用。
        system_prompt：作为 LLM 载荷位置 [1] 注入的 markdown 正文（通过
            :meth:`MemoryManager.build_llm_context`）。
    """

    character_id: str
    name: str
    avatar: str | None
    greeting: str | None
    system_prompt: str


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown text with optional YAML frontmatter into ``(meta, body)``.

    Frontmatter is delimited by two ``---`` lines. Absent opening delimiter
    means "no frontmatter" and the whole text is returned as body.

    Raises:
        ValueError: When the opening ``---`` is present but the closing
            ``---`` is missing, or when the frontmatter YAML is not a mapping.

    将带有可选 YAML frontmatter 的 markdown 文本拆分为 ``(meta, body)``。

    Frontmatter 由两行 ``---`` 分隔。若缺少起始分隔符，则视为"无 frontmatter"，
    整段文本作为 body 返回。
    """
    # Defensively strip leftover BOM in case the reader already decoded via
    # utf-8-sig but the text still contains a stray marker.
    # 防御性去掉残留 BOM（即便 reader 已用 utf-8-sig 解码，仍可能残留标记）。
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return {}, text.strip()

    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        raise ValueError(
            "Persona markdown opens with '---' but no closing '---' delimiter "
            "was found. Frontmatter block is malformed."
        )

    fm_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).strip()
    meta = yaml.safe_load(fm_text) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"Persona frontmatter must be a YAML mapping, got {type(meta).__name__}")
    return meta, body


def load_persona(character_id: str) -> Persona:
    """Load ``prompts/persona/{character_id}.md`` and return a :class:`Persona`.

    Args:
        character_id: The character identifier; also the markdown filename
            stem.

    Returns:
        A fully populated :class:`Persona`. Missing frontmatter fields default
        to ``None`` (``avatar`` / ``greeting``) or to ``character_id`` itself
        (``name``).

    Raises:
        FileNotFoundError: When ``prompts/persona/{character_id}.md`` does not
            exist. Propagated from :func:`prompts.prompt_loader.load_persona`.
        ValueError: When the frontmatter is malformed (see
            :func:`_split_frontmatter`).

    加载 ``prompts/persona/{character_id}.md`` 并返回一个 :class:`Persona`。

    参数：
        character_id：角色标识，同时也是 markdown 文件名

    返回：
        一个字段完整的 :class:`Persona`。frontmatter 缺失字段的默认值为
        ``None``（``avatar`` / ``greeting``）或 ``character_id`` 本身
        （``name``）。

    抛出：
        FileNotFoundError：当 ``prompts/persona/{character_id}.md`` 不存在时，
            由 :func:`prompts.prompt_loader.load_persona` 传播。
        ValueError：当 frontmatter 格式不合法时（见 :func:`_split_frontmatter`）。
    """
    text = _load_persona_text(character_id)
    meta, body = _split_frontmatter(text)

    name = meta.get("name", character_id)
    avatar = meta.get("avatar")
    greeting = meta.get("greeting")

    return Persona(
        character_id=character_id,
        name=str(name),
        avatar=str(avatar) if avatar is not None else None,
        greeting=str(greeting) if greeting is not None else None,
        system_prompt=body,
    )


__all__ = ["Persona", "load_persona"]
