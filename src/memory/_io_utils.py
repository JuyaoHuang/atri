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

记忆子系统共享的底层 I/O 辅助工具。

目前只暴露 :func:`atomic_replace`，这是一个对 Windows 友好的
:func:`os.replace` 包装，能够在瞬态 ``PermissionError`` 时重试。

为什么需要这个（Windows 特有问题）：

    在 Windows 上，Defender / 搜索索引器经常在我们释放文件句柄后的几微秒内
    就打开刚关闭的文件进行扫描。扫描期间，``os.replace(src, dst)`` 会返回
    ``[WinError 5] 拒绝访问``，即便我们的代码中已经没有任何代码持有该文件。
    扫描窗口很短（数十毫秒级），因此带退避的有限重试可以在不掩盖真正错误的
    前提下化解这场竞争。

    Linux / macOS 的 rename 语义不受此影响，因此那里重试是空操作
    （第一次尝试总是成功）。

短期记忆 + chat_history 的每次保存（每个会话数十次）都会用到这个函数；
如果一次保存失败，在 Windows 上将导致整个实时会话崩溃。
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

    执行 :func:`os.replace`，并在 Windows 上发生锁竞争时进行有限次重试。

    仅捕获 :class:`PermissionError` 进行重试；其他 OS 错误（源文件缺失、
    路径无效、磁盘空间满等）立即向上传播，以便真正的 bug 能被明显暴露。
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
