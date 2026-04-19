"""Short-term memory persistence (``short_term_memory.json``).

This file holds everything the Memory Manager needs to assemble the LLM
context for a character's active session: the layered compression blocks
(``meta_blocks`` -> ``active_blocks``) plus the most-recent raw rounds.

Reference: docs/记忆系统设计讨论.md §5.3.

File layout:

    data/
      characters/
        {character}/
          short_term_memory.json      <- this module
          sessions/{session_id}.json  <- chat_history.py

Writes are atomic: we serialise to ``.json.tmp`` then ``os.replace`` onto the
final path, so a crashed save never leaves a half-written primary file.

短期记忆持久化（``short_term_memory.json``）。

本文件存放 Memory Manager 为某个角色活动会话组装 LLM 上下文所需的一切：
分层压缩块（``meta_blocks`` -> ``active_blocks``）以及最近的原始轮次。

参考：docs/记忆系统设计讨论.md §5.3。

文件布局：

    data/
      characters/
        {character}/
          short_term_memory.json      <- 本模块
          sessions/{session_id}.json  <- chat_history.py

写入是原子化的：先序列化到 ``.json.tmp``，再通过 ``os.replace`` 覆盖到目标
路径，因此保存过程中崩溃也不会在主文件中留下半写状态。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.memory._io_utils import atomic_replace

_FILENAME = "short_term_memory.json"


def _now_iso_z() -> str:
    """Return the current UTC time as ISO 8601 with trailing ``Z``.

    以 ISO 8601 格式（带结尾 ``Z``）返回当前 UTC 时间。
    """
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ShortTermStore:
    """Read/write the short-term memory JSON for one character.

    One instance is scoped to a specific (character, session). Callers load
    the current state, mutate it in memory, and call ``save`` to persist.

    针对单个角色的短期记忆 JSON 读/写器。

    每个实例作用域绑定到特定的 (character, session)。调用方加载当前状态，
    在内存中修改，再调用 ``save`` 进行持久化。
    """

    def __init__(self, character_dir: Path, session_id: str, character: str) -> None:
        self.character_dir = Path(character_dir)
        self.session_id = session_id
        self.character = character
        self.character_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.character_dir / _FILENAME

    @classmethod
    def get_skeleton(cls, session_id: str = "", character: str = "") -> dict[str, Any]:
        """Return a fresh empty state dict.

        返回一个全新的空状态字典。
        """
        return {
            "session_id": session_id,
            "character": character,
            "updated_at": _now_iso_z(),
            "total_rounds": 0,
            "meta_blocks": [],
            "active_blocks": [],
            "recent_messages": [],
        }

    def load(self) -> dict[str, Any]:
        """Return the persisted state, or a fresh skeleton when the file is absent.

        返回已持久化的状态；若文件不存在则返回一个全新的空骨架。
        """
        if not self.path.exists():
            return self.get_skeleton(self.session_id, self.character)
        with self.path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected short_term payload (not a dict) at {self.path}")
        return data

    def save(self, state: dict[str, Any]) -> None:
        """Persist ``state`` atomically, stamping ``updated_at`` with now.

        以原子方式持久化 ``state``，并将 ``updated_at`` 标记为当前时间。
        """
        out = dict(state)
        out["updated_at"] = _now_iso_z()
        tmp = self.path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            atomic_replace(tmp, self.path)
        except Exception:
            # Best-effort cleanup of the partial tmp file; propagate the error.
            # 尽力清理残留的临时文件；错误继续向上传播。
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise


__all__ = ["ShortTermStore"]
