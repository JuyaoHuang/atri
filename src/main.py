"""Application entry point.

Initializes the logger and loads the root config, then logs a startup banner.
This skeleton verifies Phase 1 infrastructure end-to-end; subsequent phases
will extend ``main`` with service context assembly and FastAPI launch.

Run::

    uv run python -m src.main

Reference: docs/项目架构设计.md §2.2
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.utils.config_loader import load_config
from src.utils.logger import init_logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


def main() -> None:
    init_logger()
    logger.info("atri starting")
    config = load_config(_DEFAULT_CONFIG)
    logger.info("Config loaded | sections={}", sorted(config.keys()))


if __name__ == "__main__":
    main()
