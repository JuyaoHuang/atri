"""Tests for src/memory/chat_history.py (ChatHistoryWriter).

针对 src/memory/chat_history.py（ChatHistoryWriter）的测试。

覆盖点：``ensure_metadata`` 的幂等性（仅追加一次）、human/ai 消息中
可选字段（``raw_input`` / ``avatar``）按存在性写入、插入顺序的保真迭代、
metadata + 对话内容的往返一致性、``sessions/`` 目录自动创建、
非数组 payload 的错误检测，以及 system 消息不携带 name/avatar 字段。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.chat_history import ChatHistoryWriter


def test_ensure_metadata_only_appends_once(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "2026-04-18_abcd1234", "katou")
    w.ensure_metadata()
    w.ensure_metadata()
    w.ensure_metadata()
    msgs = list(w.iter_messages())
    metas = [m for m in msgs if m.get("role") == "metadata"]
    assert len(metas) == 1
    assert metas[0]["session_id"] == "2026-04-18_abcd1234"
    assert metas[0]["character"] == "katou"


def test_append_human_without_raw_input_omits_key(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.ensure_metadata()
    w.append_human("hello")
    msgs = list(w.iter_messages())
    human = next(m for m in msgs if m["role"] == "human")
    assert human["content"] == "hello"
    assert "raw_input" not in human
    assert human["name"] == "user"


def test_append_human_with_raw_input_includes_key(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.append_human("今天天气真好", raw_input="嗯那个今天天气真好啊", name="Alen")
    human = next(m for m in w.iter_messages() if m["role"] == "human")
    assert human["raw_input"] == "嗯那个今天天气真好啊"
    assert human["name"] == "Alen"


def test_append_ai_without_avatar_omits_key(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.append_ai("Hello!", name="Katou")
    ai = next(m for m in w.iter_messages() if m["role"] == "ai")
    assert "avatar" not in ai
    assert ai["name"] == "Katou"


def test_append_ai_with_avatar_includes_key(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.append_ai("Hi!", name="Katou", avatar="katou.png")
    ai = next(m for m in w.iter_messages() if m["role"] == "ai")
    assert ai["avatar"] == "katou.png"


def test_iter_messages_yields_in_insertion_order(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.ensure_metadata()
    w.append_human("first")
    w.append_ai("second", name="Katou")
    w.append_system("[Interrupted by user]")

    roles = [m["role"] for m in w.iter_messages()]
    assert roles == ["metadata", "human", "ai", "system"]


def test_metadata_plus_two_messages_roundtrip(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.ensure_metadata()
    w.append_human("hi")
    w.append_ai("hello", name="Katou")

    # Fresh writer should see the same data.
    # 新建的 writer 应能看到相同的数据。
    w2 = ChatHistoryWriter(tmp_path, "s1", "katou")
    msgs = list(w2.iter_messages())
    assert len(msgs) == 3
    assert msgs[0]["role"] == "metadata"
    assert msgs[1]["content"] == "hi"
    assert msgs[2]["content"] == "hello"


def test_sessions_directory_auto_created(tmp_path: Path) -> None:
    char_dir = tmp_path / "characters" / "katou"
    assert not char_dir.exists()
    _ = ChatHistoryWriter(char_dir, "s1", "katou")
    assert (char_dir / "sessions").is_dir()


def test_load_raises_on_non_list_payload(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.path.parent.mkdir(parents=True, exist_ok=True)
    w.path.write_text('{"role":"metadata"}', encoding="utf-8")
    with pytest.raises(ValueError):
        list(w.iter_messages())


def test_system_message_has_no_name_or_avatar(tmp_path: Path) -> None:
    w = ChatHistoryWriter(tmp_path, "s1", "katou")
    w.append_system("[Interrupted]")
    sys_msg = next(m for m in w.iter_messages() if m["role"] == "system")
    assert "name" not in sys_msg
    assert "avatar" not in sys_msg
    assert sys_msg["content"] == "[Interrupted]"
