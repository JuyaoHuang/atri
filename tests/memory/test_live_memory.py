"""Live end-to-end memory-system tests (mem0 SDK + real LLM).

Gated by ``@pytest.mark.live`` and excluded from the default ``pytest`` run.
To execute::

    uv run pytest tests/memory/test_live_memory.py -m live -v

Required environment variables (loaded from the project-root ``.env`` via
``python-dotenv``)::

    MEM0_API_KEY=m0-...
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://.../v1
    OPENAI_MODEL=...

Each test hits the real mem0 SaaS (``app.mem0.ai``) and the real LLM. A
single run costs the mem0 fact-extraction tokens for one 20-round window
plus one search + one build_llm_context -- enough to validate the wiring
end-to-end without being expensive. Tests are skipped if any env var is
missing or still a literal ``${VAR}`` placeholder (config-loader did not
resolve it).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from src.llm import LLMFactory, LLMInterface
from src.memory.long_term import LongTermMemory
from src.memory.manager import MemoryManager

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

_REQUIRED_ENV = ("MEM0_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")


def _require_live_env() -> dict[str, str]:
    """Return resolved env vars or pytest.skip with a clear message."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in _REQUIRED_ENV:
        value = os.environ.get(name, "")
        if not value or (value.startswith("${") and value.endswith("}")):
            missing.append(name)
        else:
            resolved[name] = value
    if missing:
        pytest.skip(f"live memory test skipped -- missing env vars: {missing}")
    return resolved


def _memory_config_sdk(mem0_api_key: str) -> dict[str, Any]:
    """Build the memory_config used by the live test, with mem0.mode forced to sdk.

    This does NOT mutate ``config/memory_config.yaml`` -- the override lives
    only for the duration of the test.
    """
    return {
        "short_term": {
            "snip": {
                "filler_words": ["嗯", "啊", "呃", "额"],
                "similarity_threshold": 0.95,
                "max_single_message_tokens": 800,
            },
            "collapse": {
                "trigger_rounds": 26,
                "compress_rounds": 20,
                "keep_recent_rounds": 6,
            },
            "super_compact": {"trigger_blocks": 4},
            "compressor": {
                "l3_role": "l3_compress",
                "l4_role": "l4_compact",
            },
        },
        "mem0": {
            "mode": "sdk",
            "sdk": {"api_key": mem0_api_key},
            "search": {"limit": 5, "threshold": 0.3, "rerank": False},
        },
        "storage": {"characters_dir": "./data/characters"},
    }


def _make_live_llm_factory(env: dict[str, str]) -> tuple[LLMInterface, Any]:
    """Create a shared LLMInterface + a factory callable that returns it."""
    llm = LLMFactory.create(
        "openai_compatible",
        model=env["OPENAI_MODEL"],
        base_url=env["OPENAI_BASE_URL"],
        api_key=env["OPENAI_API_KEY"],
        temperature=0.3,
    )

    def factory(role: str) -> LLMInterface:
        return llm

    return llm, factory


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_memory_manager_full_flow(tmp_path: Path) -> None:
    """Drive 26 rounds -> L3 fires -> mem0.add succeeds -> search + context build."""
    env = _require_live_env()
    config = _memory_config_sdk(env["MEM0_API_KEY"])

    long_term = LongTermMemory(config["mem0"])
    _, factory = _make_live_llm_factory(env)

    mgr = MemoryManager(
        config,
        factory,
        character="atri_live",
        user_id="pytest_alice",
        character_dir=tmp_path,
        long_term=long_term,
    )

    # 26 rounds of synthetic content -- varied enough that mem0 can extract
    # at least one fact from the compressed window.
    prompts = [
        "我最爱的饮品是珍珠奶茶",
        "我的宠物叫毛毛",
        "我周末喜欢弹钢琴",
        "我在杭州工作",
        "我害怕坐飞机",
    ]
    for i in range(26):
        human_content = prompts[i % len(prompts)] + f"（第{i + 1}轮）"
        ai_content = f"了解了，第{i + 1}轮我记住了：{prompts[i % len(prompts)]}"
        await mgr.on_round_complete(
            {"role": "human", "content": human_content, "name": "pytest_alice"},
            {"role": "ai", "content": ai_content, "name": "atri_live"},
        )

    assert mgr.state["total_rounds"] == 26
    assert len(mgr.state["active_blocks"]) == 1
    block = mgr.state["active_blocks"][0]
    assert block["covers_rounds"] == [1, 20]
    # L3 trigger already pushed rounds 1-20 to mem0; no exception means success.

    results = await mgr.search_long_term("favorite drink")
    assert isinstance(results, list)

    messages = await mgr.build_llm_context("好的，继续聊")
    assert len(messages) > 0
    assert messages[-1] == {"role": "user", "content": "好的，继续聊"}
    # Structure check: at least the 1 active block + 12 recent msgs + user.
    assert len(messages) >= 14

    long_term.close()
