"""Loguru-based logger initialization.

Console: ``INFO`` by default, colorized. File: ``DEBUG`` to
``logs/debug_{YYYY-MM-DD}.log`` with 10 MB rotation and 30-day retention.

基于 Loguru 的日志记录器初始化。

控制台：默认 ``INFO`` 级别，带颜色输出。文件：``DEBUG`` 级别输出到
``logs/debug_{YYYY-MM-DD}.log``，10 MB 轮转、30 天保留期。

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

    使用控制台 + 文件 sink 配置 loguru。

    删除所有已存在的处理器，以便重复调用产生干净的配置。文件 sink 始终以
    DEBUG 级别记录，不受控制台级别影响——完整保真度的日志写入磁盘，控制台
    只显示过滤后的视图。

    参数：
        console_log_level：控制台 sink 的日志级别。可选值：
            ``"TRACE" | "DEBUG" | "INFO" | "SUCCESS" | "WARNING" | "ERROR"
            | "CRITICAL"``。默认 ``"INFO"``。
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
