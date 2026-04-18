"""Tests for src.utils.logger.

测试目的：验证 loguru 日志记录器初始化的正确性，包括默认调用不抛出异常、
接受自定义控制台日志级别，以及能够按预期在 ``logs/`` 目录下创建
``debug_*.log`` 日志文件。
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from loguru import logger

from src.utils.logger import init_logger


@pytest.fixture(autouse=True)
def _reset_logger() -> Generator[None, None, None]:
    """Detach all sinks after each test to avoid bleed-over state."""
    yield
    logger.remove()


def test_init_logger_does_not_raise() -> None:
    init_logger()
    logger.info("smoke test message")


def test_init_logger_accepts_custom_console_level() -> None:
    init_logger(console_log_level="DEBUG")
    logger.debug("debug level message")


def test_init_logger_creates_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_logger()
    logger.info("triggers log file creation")
    logger.complete()

    logs_dir = tmp_path / "logs"
    assert logs_dir.is_dir(), "logs/ directory should be created by the file sink"
    assert list(logs_dir.glob("debug_*.log")), "at least one debug log file should exist"
