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
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SESSIONS_SUBDIR = "sessions"


def _now_iso_z() -> str:
    """Return the current UTC time as ISO 8601 with trailing ``Z``."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ChatHistoryWriter:
    """Append-only writer for one session's chat history JSON."""

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
    # ------------------------------------------------------------------

    def ensure_metadata(self) -> None:
        """Append the metadata row iff it is not already present.

        Safe to call multiple times during session lifecycle (start / resume).
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
        (its absence signals a text-typed input per §5.2)."""
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
        """Append an AI message. ``avatar`` is included only when provided."""
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
        """Append a system-level marker (interruption notices, etc.)."""
        self._append(
            {
                "role": "system",
                "timestamp": _now_iso_z(),
                "content": content,
            }
        )

    def iter_messages(self) -> Iterator[dict[str, Any]]:
        """Yield every stored message in insertion order."""
        yield from self._load_array()

    # ------------------------------------------------------------------
    # Internal: atomic parse-modify-rewrite
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
            os.replace(tmp, self.path)
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


__all__ = ["ChatHistoryWriter"]
