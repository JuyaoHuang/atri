"""Tests for src/agent/persona.py (Persona dataclass + load_persona).

Covers Phase 4 PRD US-AGT-001 acceptance criteria:
  (a) load_persona('atri') returns a Persona with all frontmatter fields
  (b) character_id comes from the function arg, not the frontmatter
  (c) missing-optional fields (avatar / greeting) default to None
  (d) body preserves markdown structure verbatim
  (e) missing file raises FileNotFoundError with the attempted path
  (f) malformed frontmatter raises ValueError
  (g) UTF-8 with BOM is tolerated

针对 src/agent/persona.py 的测试（Persona dataclass + load_persona）。

覆盖 Phase 4 PRD US-AGT-001 验收标准：
  (a) load_persona('atri') 返回字段完整的 Persona
  (b) character_id 来自函数参数，而非 frontmatter
  (c) 缺失的可选字段（avatar / greeting）默认为 None
  (d) body 原样保留 markdown 结构
  (e) 缺失文件抛出 FileNotFoundError 含尝试路径
  (f) frontmatter 非法时抛出 ValueError
  (g) 容忍 UTF-8 with BOM
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prompts import prompt_loader
from src.agent.persona import Persona, _split_frontmatter, load_persona

# ---------------------------------------------------------------------------
# _split_frontmatter unit tests (pure parser, no IO)
# _split_frontmatter 的纯解析器单元测试（无 IO）
# ---------------------------------------------------------------------------


def test_split_frontmatter_parses_full_header() -> None:
    text = """---
name: 亚托莉
avatar: atri.png
---

# 角色
你是 ATRI。"""
    meta, body = _split_frontmatter(text)
    assert meta == {"name": "亚托莉", "avatar": "atri.png"}
    assert body.startswith("# 角色")
    assert "你是 ATRI。" in body


def test_split_frontmatter_returns_empty_dict_when_no_delimiter() -> None:
    text = "# 只是 markdown\n没有 frontmatter。"
    meta, body = _split_frontmatter(text)
    assert meta == {}
    assert body == "# 只是 markdown\n没有 frontmatter。"


def test_split_frontmatter_missing_closing_raises() -> None:
    text = "---\nname: atri\n# body without closing"
    with pytest.raises(ValueError, match="closing"):
        _split_frontmatter(text)


def test_split_frontmatter_tolerates_leading_bom() -> None:
    text = "\ufeff---\nname: atri\n---\n\nbody"
    meta, body = _split_frontmatter(text)
    assert meta == {"name": "atri"}
    assert body == "body"


def test_split_frontmatter_rejects_non_mapping_yaml() -> None:
    text = "---\n- list\n- values\n---\n\nbody"
    with pytest.raises(ValueError, match="mapping"):
        _split_frontmatter(text)


# ---------------------------------------------------------------------------
# load_persona integration tests
# load_persona 的集成测试
# ---------------------------------------------------------------------------


def test_load_persona_returns_populated_persona_for_atri() -> None:
    """Shipped prompts/persona/atri.md parses into a full Persona.

    Phase 4 ships atri.md; this end-to-end test pins the real file.

    Phase 4 随仓库发布 atri.md；此端到端测试钉住真实文件。
    """
    p = load_persona("atri")
    assert isinstance(p, Persona)
    assert p.character_id == "atri"
    assert p.name == "亚托莉"
    assert p.avatar == "atri.png"
    assert p.greeting is not None and len(p.greeting) > 0
    assert "角色设定" in p.system_prompt
    assert "高性能" in p.system_prompt


def test_load_persona_character_id_from_arg_not_frontmatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """character_id comes from function arg even if frontmatter tries to override.

    即便 frontmatter 试图覆盖，character_id 仍来自函数参数。
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "bob.md").write_text(
        "---\nname: 真名\ncharacter_id: should_be_ignored\n---\n\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_loader, "_PERSONA_DIR", persona_dir)

    p = load_persona("bob")
    assert p.character_id == "bob"
    assert p.name == "真名"


def test_load_persona_optional_fields_default_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent avatar / greeting default to None; absent name defaults to character_id.

    缺失 avatar / greeting 默认为 None；缺失 name 回退为 character_id。
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "min.md").write_text(
        "---\nname: Minimalist\n---\n\nhello\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_loader, "_PERSONA_DIR", persona_dir)

    p = load_persona("min")
    assert p.name == "Minimalist"
    assert p.avatar is None
    assert p.greeting is None
    assert p.system_prompt == "hello"


def test_load_persona_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    monkeypatch.setattr(prompt_loader, "_PERSONA_DIR", persona_dir)

    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_persona("nonexistent")


def test_load_persona_tolerates_utf8_bom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persona file saved with UTF-8 BOM parses cleanly (consistent with load_compress).

    持 UTF-8 BOM 的 persona 文件能正常解析（与 load_compress 行为一致）。
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "bom.md").write_text(
        "---\nname: Bomee\n---\n\nbody text\n",
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(prompt_loader, "_PERSONA_DIR", persona_dir)

    p = load_persona("bom")
    assert p.name == "Bomee"
    assert p.system_prompt == "body text"


def test_load_persona_body_preserves_markdown_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Headers / lists / paragraphs in body survive the split verbatim.

    body 中的标题 / 列表 / 段落在拆分后原样保留。
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "rich.md").write_text(
        """---
name: Rich
---

# Heading 1

- list item a
- list item b

## Heading 2

Paragraph text.
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_loader, "_PERSONA_DIR", persona_dir)

    p = load_persona("rich")
    assert "# Heading 1" in p.system_prompt
    assert "- list item a" in p.system_prompt
    assert "- list item b" in p.system_prompt
    assert "## Heading 2" in p.system_prompt
    assert "Paragraph text." in p.system_prompt


def test_load_persona_no_frontmatter_defaults_name_to_character_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persona file without any frontmatter still loads; name = character_id.

    完全没有 frontmatter 的 persona 文件仍可加载；name 回退为 character_id。
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "bare.md").write_text(
        "# Just a body\n\nNo frontmatter here.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_loader, "_PERSONA_DIR", persona_dir)

    p = load_persona("bare")
    assert p.name == "bare"
    assert p.avatar is None
    assert p.greeting is None
    assert "Just a body" in p.system_prompt
