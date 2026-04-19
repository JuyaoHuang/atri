"""Application entry point.

Initializes the logger, loads the root config, then resolves each of the
three configured LLM call-sites (``chat``, ``l3_compress``, ``l4_compact``)
via the role-based factory. Instantiating the LLM clients does not touch
the network -- only construction; actual requests happen later in the
agent/memory layers.

Startup also wires a :class:`MemoryManager` so we can verify the memory
subsystem constructs cleanly. Long-term (:class:`LongTermMemory`)
construction is best-effort: in ``local_deploy`` mode it may hit
Qdrant/Ollama, so any failure downgrades to a WARNING and the manager
runs without long-term context (short-term compression still works).

Run::

    uv run python -m src.main

应用程序入口点。

初始化日志记录器，加载根配置，然后通过基于角色的工厂解析三个已配置的
LLM 调用位点 (``chat``、``l3_compress``、``l4_compact``) 中的每一个。
实例化 LLM 客户端不会触发网络请求——仅执行构造；实际请求稍后在
agent/memory 层中发生。

运行方式::

    uv run python -m src.main

Reference: docs/项目架构设计.md §2.2, docs/LLM调用层设计讨论.md §2.3,
docs/记忆系统设计讨论.md §6.1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from src.llm import create_from_role
from src.llm.interface import LLMInterface
from src.memory.long_term import LongTermMemory
from src.memory.manager import MemoryManager
from src.utils.config_loader import load_config
from src.utils.logger import init_logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"
_DEFAULT_ENV = _REPO_ROOT / ".env"

_LLM_ROLES = ("chat", "l3_compress", "l4_compact")
_DEMO_CHARACTER = "atri"
_DEMO_USER_ID = "default"


def _make_llm_factory_fn(llm_config: dict[str, Any]):
    """Return a ``role -> LLMInterface`` closure for :class:`MemoryManager`.

    返回一个供 :class:`MemoryManager` 使用的 ``role -> LLMInterface`` 闭包。
    """

    def factory(role: str) -> LLMInterface:
        return create_from_role(role, llm_config)

    return factory


def _safe_build_long_term(mem0_config: dict[str, Any]) -> LongTermMemory | None:
    """Best-effort :class:`LongTermMemory` construction.

    In ``local_deploy`` mode, ``Memory.from_config`` may ping Qdrant/Ollama
    as part of initialization -- if those backends are not running during a
    cold start, we log a WARNING and return ``None`` so the rest of the
    startup can proceed. Short-term compression still works fully.

    尽力而为地构造 :class:`LongTermMemory`。

    在 ``local_deploy`` 模式下，``Memory.from_config`` 初始化时可能会访问
    Qdrant/Ollama——如果在冷启动阶段这些后端尚未运行，我们记录 WARNING 并
    返回 ``None``，以便后续启动流程继续推进。短期压缩仍可完整工作。
    """
    if not mem0_config:
        logger.warning("mem0 config missing; skipping LongTermMemory construction")
        return None
    try:
        return LongTermMemory(mem0_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LongTermMemory construction failed (continuing without long-term "
            "memory) | mode={} error={!r}",
            mem0_config.get("mode"),
            exc,
        )
        return None


def main() -> None:
    init_logger()
    logger.info("atri starting")

    dotenv_loaded = load_dotenv(_DEFAULT_ENV)
    logger.info("Dotenv loaded | path={} present={}", _DEFAULT_ENV, dotenv_loaded)

    config = load_config(_DEFAULT_CONFIG)
    logger.info("Config loaded | sections={}", sorted(config.keys()))

    llm_config = config.get("llm", {})
    for role in _LLM_ROLES:
        try:
            llm = create_from_role(role, llm_config)
        except (KeyError, ValueError) as exc:
            logger.error("LLM role failed | role={} error={}", role, exc)
            continue
        logger.info(
            "LLM role resolved | role={} provider={} model={}",
            role,
            type(llm).__name__,
            getattr(llm, "model", "n/a"),
        )

    memory_config = config.get("memory", {})
    mem0_cfg = memory_config.get("mem0", {})
    mem0_mode = mem0_cfg.get("mode", "local_deploy")

    long_term = _safe_build_long_term(mem0_cfg)

    try:
        MemoryManager(
            memory_config,
            _make_llm_factory_fn(llm_config),
            character=_DEMO_CHARACTER,
            user_id=_DEMO_USER_ID,
            long_term=long_term,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("MemoryManager construction failed | error={!r}", exc)
        return

    logger.info(
        "MemoryManager ready | mode={} | character={} | long_term={}",
        mem0_mode,
        _DEMO_CHARACTER,
        "on" if long_term is not None else "off",
    )


if __name__ == "__main__":
    main()
