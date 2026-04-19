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
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.memory._io_utils import atomic_replace

_FILENAME = "short_term_memory.json"


def _now_iso_z() -> str:
    """Return the current UTC time as ISO 8601 with trailing ``Z``."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ShortTermStore:
    """Read/write the short-term memory JSON for one character.

    One instance is scoped to a specific (character, session). Callers load
    the current state, mutate it in memory, and call ``save`` to persist.
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
        """Return a fresh empty state dict."""
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
        """Return the persisted state, or a fresh skeleton when the file is absent."""
        if not self.path.exists():
            return self.get_skeleton(self.session_id, self.character)
        with self.path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected short_term payload (not a dict) at {self.path}")
        return data

    def save(self, state: dict[str, Any]) -> None:
        """Persist ``state`` atomically, stamping ``updated_at`` with now."""
        out = dict(state)
        out["updated_at"] = _now_iso_z()
        tmp = self.path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            atomic_replace(tmp, self.path)
        except Exception:
            # Best-effort cleanup of the partial tmp file; propagate the error.
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise


__all__ = ["ShortTermStore"]
