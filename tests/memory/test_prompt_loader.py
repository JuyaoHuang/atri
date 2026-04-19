"""Tests for prompts/prompt_loader.py.

针对 prompts/prompt_loader.py 的测试。

覆盖点：L3 压缩提示模板的非空性与关键标签（``<analysis>``、P0-P3 优先级）
存在性、未知名称查找时的 FileNotFoundError（compress/persona/utils 三个分支），
以及对手工编辑文件可能引入的 UTF-8 BOM 的透明剥离。
"""

from __future__ import annotations

import pytest

from prompts.prompt_loader import load_compress, load_persona, load_util


def test_load_compress_l3_returns_non_empty_template_with_analysis_tag() -> None:
    text = load_compress("l3_collapse")
    assert text.strip() != ""
    assert "<analysis>" in text
    # Priority tags from §8.6 must be preserved.
    # §8.6 中定义的优先级标签必须被保留。
    for prio in ("P0", "P1", "P2", "P3"):
        assert prio in text


def test_load_compress_missing_name_raises() -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_compress("this_file_does_not_exist_in_compress")
    # Error message must reveal the attempted path for easy debugging.
    # 错误信息必须暴露尝试的路径，便于调试。
    assert "compress" in str(exc.value)
    assert "this_file_does_not_exist_in_compress" in str(exc.value)


def test_load_persona_missing_name_raises() -> None:
    # Phase 3 does not populate persona/, so any lookup should fail cleanly.
    # Phase 3 不填充 persona/ 目录，因此任何查找都应干净地失败。
    with pytest.raises(FileNotFoundError) as exc:
        load_persona("nonexistent_persona")
    assert "persona" in str(exc.value)


def test_load_util_missing_name_raises() -> None:
    # Phase 3 does not populate utils/ either.
    # Phase 3 同样不填充 utils/ 目录。
    with pytest.raises(FileNotFoundError) as exc:
        load_util("nonexistent_util")
    assert "utils" in str(exc.value)


def test_load_compress_handles_utf8_bom(tmp_path, monkeypatch) -> None:
    """Regression guard: BOM in hand-edited prompt files must not leak through.

    回归保障：手工编辑的提示文件中的 BOM 不应透传到加载结果。
    """
    import prompts.prompt_loader as pl

    fake_compress = tmp_path / "compress"
    fake_compress.mkdir()
    content = "hello-bom"
    # Write with BOM.
    # 写入带 BOM 的内容。
    (fake_compress / "bomtest.txt").write_bytes("\ufeff".encode("utf-8") + content.encode())

    monkeypatch.setattr(pl, "_COMPRESS_DIR", fake_compress)
    assert pl.load_compress("bomtest") == content
