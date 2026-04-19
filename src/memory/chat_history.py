"""Chat history persistence (``sessions/{session_id}.json``).

Append-only archive of every raw message the user and AI exchange within a
single session. The frontend renders from this file; the backend also uses
it as the source-of-truth for §8.5 session-resume consistency checks.

File format (per design doc §5.2): a JSON array of objects, where the first
entry is a ``metadata`` row carrying session/character identifiers. Each
subsequent entry is one ``human`` / ``ai`` / ``system`` message.

Writes are atomic: parse-modify-rewrite via a ``.tmp`` file + ``os.replace``.
For expected session sizes (hundreds to low thousands of messages) this is
fast enough and keeps the file human-readable / recoverable.

Reading is *tolerant*: :meth:`ChatHistoryWriter.iter_messages` recovers the
parseable prefix when the file has a trailing malformed record, logging a
WARNING rather than propagating :class:`json.JSONDecodeError`. This keeps
US-MEM-008 ``resume_session`` robust when a partial write was interrupted.

聊天历史持久化（``sessions/{session_id}.json``）。

单个会话内，用户和 AI 交换的每条原始消息的仅追加归档。前端从此文件渲染；
后端也将其作为 §8.5 会话恢复一致性校验的权威来源。

文件格式（参见设计文档 §5.2）：一个 JSON 对象数组，其中首条为 ``metadata``
行，携带会话/角色标识；之后每条为一个 ``human`` / ``ai`` / ``system`` 消息。

写入是原子化的：通过 ``.tmp`` 文件 + ``os.replace`` 进行解析-修改-重写。
对于预期的会话体量（数百到数千条消息）足够快，并且文件保持人类可读 / 可恢复。

读取是 *容错* 的：当文件尾部存在畸形记录时，
:meth:`ChatHistoryWriter.iter_messages` 会恢复可解析的前缀并记录 WARNING，
而非传播 :class:`json.JSONDecodeError`。这让 US-MEM-008 ``resume_session``
在写入中断时仍保持稳健。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.memory._io_utils import atomic_replace

_SESSIONS_SUBDIR = "sessions"


def _now_iso_z() -> str:
    """Return the current UTC time as ISO 8601 with trailing ``Z``.

    以 ISO 8601 格式（带结尾 ``Z``）返回当前 UTC 时间。
    """
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ChatHistoryWriter:
    """Append-only writer for one session's chat history JSON.

    针对单个会话聊天历史 JSON 的仅追加写入器。
    """

    def __init__(self, character_dir: Path, session_id: str, character: str) -> None:
        self.character_dir = Path(character_dir)
        self.session_id = session_id
        self.character = character
        (self.character_dir / _SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.character_dir / _SESSIONS_SUBDIR / f"{self.session_id}.json"

    # ------------------------------------------------------------------
    # Public append API
    # 公共追加 API
    # ------------------------------------------------------------------

    def ensure_metadata(self) -> None:
        """Append the metadata row iff it is not already present.

        Safe to call multiple times during session lifecycle (start / resume).

        仅当 metadata 行尚未存在时追加。

        在会话生命周期中（启动 / 恢复）可安全地多次调用。
        """
        data = self._load_array()
        if any(m.get("role") == "metadata" for m in data):
            return
        meta = {
            "role": "metadata",
            "timestamp": _now_iso_z(),
            "session_id": self.session_id,
            "character": self.character,
        }
        data.insert(0, meta)
        self._save_array(data)

    def append_human(
        self,
        content: str,
        raw_input: str | None = None,
        name: str = "user",
    ) -> None:
        """Append a human message. ``raw_input`` is included only when provided
        (its absence signals a text-typed input per §5.2).

        追加一条 human 消息。仅在提供时才包含 ``raw_input``
        （未提供表示文本输入，参见 §5.2）。
        """
        entry: dict[str, Any] = {
            "role": "human",
            "timestamp": _now_iso_z(),
            "content": content,
            "name": name,
        }
        if raw_input is not None:
            entry["raw_input"] = raw_input
        self._append(entry)

    def append_ai(
        self,
        content: str,
        name: str,
        avatar: str | None = None,
    ) -> None:
        """Append an AI message. ``avatar`` is included only when provided.

        追加一条 AI 消息。仅在提供时才包含 ``avatar``。
        """
        entry: dict[str, Any] = {
            "role": "ai",
            "timestamp": _now_iso_z(),
            "content": content,
            "name": name,
        }
        if avatar is not None:
            entry["avatar"] = avatar
        self._append(entry)

    def append_system(self, content: str) -> None:
        """Append a system-level marker (interruption notices, etc.).

        追加一条系统级标记（中断通知等）。
        """
        self._append(
            {
                "role": "system",
                "timestamp": _now_iso_z(),
                "content": content,
            }
        )

    def iter_messages(self) -> Iterator[dict[str, Any]]:
        """Yield every stored message in insertion order.

        When the underlying JSON array has a trailing malformed record
        (partial write interrupted, disk corruption, etc.), iteration logs
        a WARNING and yields the parseable prefix instead of raising. This
        is what makes §8.5 ``resume_session`` robust enough to rebuild from
        imperfect input. Structural issues (not-a-list payload) still raise
        :class:`ValueError` so callers can treat them as a hard error.

        按插入顺序产出每一条存储的消息。

        当底层 JSON 数组存在尾部畸形记录时（部分写入被中断、磁盘损坏等），
        迭代会记录 WARNING 并产出可解析的前缀而非抛出异常。这正是让 §8.5
        ``resume_session`` 能在不完美输入下稳健重建的关键。结构性问题
        （载荷并非列表）仍会抛出 :class:`ValueError`，以便调用方将其视为
        硬错误处理。
        """
        try:
            data = self._load_array()
        except json.JSONDecodeError as exc:
            logger.warning(
                f"chat_history JSON malformed at {self.path}: {exc!r}; "
                "falling back to tolerant parse"
            )
            data = self._tolerant_parse()
        yield from data

    # ------------------------------------------------------------------
    # Internal: atomic parse-modify-rewrite
    # 内部实现：原子化的解析-修改-重写
    # ------------------------------------------------------------------

    def _load_array(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected chat_history payload (not a list) at {self.path}")
        return data

    def _save_array(self, data: list[dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            atomic_replace(tmp, self.path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise

    def _append(self, entry: dict[str, Any]) -> None:
        data = self._load_array()
        data.append(entry)
        self._save_array(data)

    def _tolerant_parse(self) -> list[dict[str, Any]]:
        """Best-effort object-by-object recovery of a JSON array.

        Walks the file text with :class:`json.JSONDecoder.raw_decode`,
        stopping at the first unparseable token. Returns whatever prefix
        was decoded successfully.

        对 JSON 数组进行尽力而为的逐对象恢复。

        用 :class:`json.JSONDecoder.raw_decode` 扫描文件文本，在遇到第一个
        无法解析的 token 处停止。返回所有成功解码的前缀。
        """
        if not self.path.exists():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        stripped = text.strip()
        if not stripped.startswith("["):
            return []
        decoder = json.JSONDecoder()
        results: list[dict[str, Any]] = []
        idx = 1  # skip leading '['
        # 跳过开头的 '['
        n = len(stripped)
        while idx < n:
            while idx < n and stripped[idx] in " \t\r\n,":
                idx += 1
            if idx >= n or stripped[idx] == "]":
                break
            try:
                obj, consumed = decoder.raw_decode(stripped, idx)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                results.append(obj)
            idx = consumed
        return results


__all__ = ["ChatHistoryWriter"]
