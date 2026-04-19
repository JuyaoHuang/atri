"""Shared low-level I/O helpers for the memory subsystem.

Currently only exposes :func:`atomic_replace`, a Windows-friendly wrapper
around :func:`os.replace` that retries on transient ``PermissionError``.

Why this exists (Windows specifics):

    On Windows, Defender / the search indexer frequently opens a freshly
    closed file for scanning microseconds after we release the handle.
    While that scan runs, ``os.replace(src, dst)`` returns ``[WinError 5]
    Access is denied`` even though nothing in our code still holds the
    file. The scan window is short (low tens of ms), so a bounded retry
    with back-off dissolves the race without masking a genuine fault.

    Linux / macOS rename semantics are immune to this, so the retry is a
    no-op there (first attempt always succeeds).

We use this for every short-term memory + chat_history save (dozens per
session); a single save failure would otherwise crash an entire live
session on Windows.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF_SEC = 0.02


def atomic_replace(src: Path, dst: Path) -> None:
    """Run :func:`os.replace` with a bounded retry on Windows lock contention.

    The retry only catches :class:`PermissionError`; other OS errors
    (missing source, invalid path, disk full) propagate immediately so
    real bugs surface loud.
    """
    last_err: PermissionError | None = None
    backoff = _INITIAL_BACKOFF_SEC
    for attempt in range(_MAX_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError as err:
            last_err = err
            if attempt == _MAX_ATTEMPTS - 1:
                break
            time.sleep(backoff)
            backoff *= 2
    assert last_err is not None
    raise last_err


__all__ = ["atomic_replace"]
