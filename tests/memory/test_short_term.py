"""Tests for src/memory/short_term.py (ShortTermStore)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import pytest

from src.memory.short_term import ShortTermStore


def test_load_on_missing_file_returns_skeleton(tmp_path: Path) -> None:
    store = ShortTermStore(tmp_path, "s1", "katou")
    state = store.load()
    assert state["session_id"] == "s1"
    assert state["character"] == "katou"
    assert state["total_rounds"] == 0
    assert state["meta_blocks"] == []
    assert state["active_blocks"] == []
    assert state["recent_messages"] == []
    assert state["updated_at"].endswith("Z")


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    store = ShortTermStore(tmp_path, "s1", "katou")
    state = ShortTermStore.get_skeleton("s1", "katou")
    state["total_rounds"] = 7
    state["active_blocks"] = [{"block_id": "block_abcd1234"}]
    state["recent_messages"] = [
        {"round": 7, "role": "human", "content": "hi"},
    ]
    store.save(state)

    loaded = store.load()
    assert loaded["total_rounds"] == 7
    assert loaded["active_blocks"][0]["block_id"] == "block_abcd1234"
    assert loaded["recent_messages"][0]["content"] == "hi"


def test_save_updates_updated_at_timestamp(tmp_path: Path) -> None:
    store = ShortTermStore(tmp_path, "s1", "katou")
    state = ShortTermStore.get_skeleton("s1", "katou")

    store.save(state)
    first_ts = store.load()["updated_at"]

    # Sleep 1.1s to cross a second boundary (iso timespec=seconds).
    time.sleep(1.1)
    store.save(state)
    second_ts = store.load()["updated_at"]

    assert second_ts >= first_ts
    assert second_ts != first_ts


def test_skeleton_classmethod_returns_clean_structure() -> None:
    skel = ShortTermStore.get_skeleton("sess-42", "shizuku")
    assert skel["session_id"] == "sess-42"
    assert skel["character"] == "shizuku"
    assert skel["total_rounds"] == 0
    # Lists must be fresh per call (no mutable default aliasing).
    skel["active_blocks"].append({"block_id": "x"})
    skel2 = ShortTermStore.get_skeleton("sess-42", "shizuku")
    assert skel2["active_blocks"] == []


def test_directory_auto_created(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "characters" / "katou"
    assert not nested.exists()
    _ = ShortTermStore(nested, "s1", "katou")
    assert nested.is_dir()


def test_save_failure_leaves_previous_content_intact(tmp_path: Path) -> None:
    store = ShortTermStore(tmp_path, "s1", "katou")
    good = ShortTermStore.get_skeleton("s1", "katou")
    good["total_rounds"] = 5
    store.save(good)
    saved_bytes = store.path.read_bytes()

    # Force json.dump to fail while writing the tmp file.
    with mock.patch.object(json, "dump", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            store.save(ShortTermStore.get_skeleton("s1", "katou"))

    # Primary file untouched.
    assert store.path.read_bytes() == saved_bytes
    # Tmp file cleaned up.
    tmp = store.path.with_suffix(".json.tmp")
    assert not tmp.exists()


def test_load_raises_on_non_dict_payload(tmp_path: Path) -> None:
    store = ShortTermStore(tmp_path, "s1", "katou")
    store.path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()
