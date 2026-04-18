"""Loguru-based logger initialization.

Console: ``INFO`` by default, colorized. File: ``DEBUG`` to
``logs/debug_{YYYY-MM-DD}.log`` with 10 MB rotation and 30-day retention.

Reference: docs/项目架构设计.md §3.2
"""

from __future__ import annotations

import sys

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{message}"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}"
)


def init_logger(console_log_level: str = "INFO") -> None:
    """Configure loguru with console + file sinks.

    Removes any pre-existing handlers so repeated calls yield a clean
    configuration. The file sink always captures at DEBUG regardless of
    the console level -- full-fidelity logs go to disk, the console shows
    a filtered view.

    Args:
        console_log_level: Log level for the console sink. One of
            ``"TRACE" | "DEBUG" | "INFO" | "SUCCESS" | "WARNING" | "ERROR"
            | "CRITICAL"``. Default ``"INFO"``.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=console_log_level,
        format=_CONSOLE_FORMAT,
        colorize=True,
    )

    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format=_FILE_FORMAT,
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )
