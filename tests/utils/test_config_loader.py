"""Tests for src.utils.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config_loader import load_config


@pytest.fixture
def config_tree(tmp_path: Path) -> Path:
    """Create root + 2 sub-configs for merge/substitution tests."""
    sub_dir = tmp_path / "config"
    sub_dir.mkdir()
    (sub_dir / "llm_config.yaml").write_text(
        "model: gpt-4o\napi_key: ${TEST_API_KEY}\n",
        encoding="utf-8",
    )
    (sub_dir / "memory_config.yaml").write_text(
        "storage:\n  path: ./data\n",
        encoding="utf-8",
    )
    root = tmp_path / "config.yaml"
    root.write_text(
        "llm_config: config/llm_config.yaml\nmemory_config: config/memory_config.yaml\n",
        encoding="utf-8",
    )
    return root


def test_merges_subfiles_under_stripped_stem(config_tree: Path) -> None:
    result = load_config(config_tree)
    assert set(result.keys()) == {"llm", "memory"}
    assert result["llm"]["model"] == "gpt-4o"
    assert result["memory"]["storage"]["path"] == "./data"


def test_env_var_substituted_when_set(config_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-secret")
    result = load_config(config_tree)
    assert result["llm"]["api_key"] == "sk-secret"


def test_env_var_preserved_when_unset(config_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    result = load_config(config_tree)
    assert result["llm"]["api_key"] == "${TEST_API_KEY}"


def test_env_var_substitution_recurses_into_lists_and_dicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("A", "alpha")
    monkeypatch.setenv("B", "bravo")
    root = tmp_path / "config.yaml"
    root.write_text(
        'flat: "${A}"\nnested:\n  key: "${B}"\nlist:\n  - "${A}"\n  - plain\n',
        encoding="utf-8",
    )
    result = load_config(root)
    assert result["flat"] == "alpha"
    assert result["nested"]["key"] == "bravo"
    assert result["list"] == ["alpha", "plain"]


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Root config"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_missing_subfile_raises(tmp_path: Path) -> None:
    root = tmp_path / "config.yaml"
    root.write_text("llm_config: config/missing.yaml\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Sub-config"):
        load_config(root)


def test_non_config_keys_pass_through(tmp_path: Path) -> None:
    root = tmp_path / "config.yaml"
    root.write_text(
        'version: "1.0"\ndebug: true\n',
        encoding="utf-8",
    )
    result = load_config(root)
    assert result == {"version": "1.0", "debug": True}
