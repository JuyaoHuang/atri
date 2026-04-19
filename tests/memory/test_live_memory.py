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

在线端到端记忆系统测试（mem0 SDK + 真实 LLM）。

由 ``@pytest.mark.live`` 标记门控，默认 ``pytest`` 运行时会被排除。
执行方式::

    uv run pytest tests/memory/test_live_memory.py -m live -v

所需环境变量（通过 ``python-dotenv`` 从项目根目录的 ``.env`` 加载）::

    MEM0_API_KEY=m0-...
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://.../v1
    OPENAI_MODEL=...

每个测试都会访问真实的 mem0 SaaS (``app.mem0.ai``) 和真实 LLM。单次运行
消耗的是 mem0 对一个 20 轮窗口的事实抽取 token，加上一次 search + 一次
build_llm_context——足以端到端验证连线路径且成本不高。当任一环境变量
缺失或仍为字面 ``${VAR}`` 占位符（配置加载器未解析）时，测试会被跳过。
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
    """Return resolved env vars or pytest.skip with a clear message.

    返回解析后的环境变量，否则给出清晰信息后 pytest.skip。
    """
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

    构建在线测试使用的 memory_config，将 mem0.mode 强制为 sdk。

    此操作不会修改 ``config/memory_config.yaml``——覆盖仅在测试期间生效。
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
    """Create a shared LLMInterface + a factory callable that returns it.

    创建共享的 LLMInterface + 返回该实例的工厂可调用对象。
    """
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
    """Drive 26 rounds -> L3 fires -> mem0.add succeeds -> search + context build.

    驱动 26 轮 -> L3 触发 -> mem0.add 成功 -> search + 上下文构建。
    """
    env = _require_live_env()
    config = _memory_config_sdk(env["MEM0_API_KEY"])

    long_term = LongTermMemory(config["mem0"])
    live_llm, factory = _make_live_llm_factory(env)

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
    # 26 轮合成内容——足够多样，使 mem0 能从压缩窗口中提取至少一个事实。
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
    # L3 触发已将第 1-20 轮推送到 mem0；无异常即视为成功。

    results = await mgr.search_long_term("favorite drink")
    assert isinstance(results, list)

    messages = await mgr.build_llm_context("好的，继续聊")
    assert len(messages) > 0
    assert messages[-1] == {"role": "user", "content": "好的，继续聊"}
    # Structure check: at least the 1 active block + 12 recent msgs + user.
    # 结构检查：至少 1 个 active block + 12 条 recent 消息 + user。
    assert len(messages) >= 14

    # Recall probes -- drive the real LLM on top of the built context so we
    # validate that the compressed L3 summary + mem0 facts + recent tail
    # together give the model enough signal to recall concrete facts from
    # earlier rounds. Without probes, the test only proves the pipeline
    # wires up; probes prove memory actually works end-to-end.
    # 召回探针——在已构建的上下文之上驱动真实 LLM，以验证压缩后的 L3 摘要
    # + mem0 事实 + 最近尾部，三者合力能否为模型提供足够信号来召回早期
    # 轮次的具体事实。没有探针时，测试只能证明流水线已连线；探针才能
    # 证明记忆系统端到端真正起作用。
    persona = (
        "你是情感陪伴 AI atri。请基于你记得的关于用户的事实，直接、简短地回答。"
        "不要编造；如果记得就说出来，不记得就坦诚说不记得。"
    )

    probe_drink = await mgr.build_llm_context(
        "你还记得我最爱什么饮品吗？",
        system_prompt=persona,
    )
    reply_drink = await live_llm.chat_completion(probe_drink)
    print(f"\n[recall probe - drink] {reply_drink!r}")
    assert "奶茶" in reply_drink, (
        f"LLM failed to recall the favorite drink (expected '奶茶'); got: {reply_drink!r}"
    )

    probe_pet = await mgr.build_llm_context(
        "我之前说过我的宠物叫什么名字？",
        system_prompt=persona,
    )
    reply_pet = await live_llm.chat_completion(probe_pet)
    print(f"[recall probe - pet] {reply_pet!r}")
    assert "毛毛" in reply_pet, (
        f"LLM failed to recall the pet name (expected '毛毛'); got: {reply_pet!r}"
    )

    long_term.close()
