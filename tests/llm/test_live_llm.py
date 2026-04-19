"""Live end-to-end tests against a real LLM API.

These tests are gated by ``@pytest.mark.live`` and excluded from the
default ``pytest`` run. To execute them::

    uv run pytest tests/llm/test_live_llm.py -m live -v

Required environment variables (loaded from the project-root ``.env`` via
``python-dotenv``; both process env and .env file are supported)::

    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...
    OPENAI_MODEL=...

Each test makes a real network call and therefore costs API tokens. The
fixture skips with a clear message when any variable is missing so the
default-run behavior is never broken.

针对真实 LLM API 的在线端到端测试。

这些测试由 ``@pytest.mark.live`` 标记门控，默认 ``pytest`` 运行时会被排除。
执行方式::

    uv run pytest tests/llm/test_live_llm.py -m live -v

所需环境变量（通过 ``python-dotenv`` 从项目根目录的 ``.env`` 加载；
同时支持进程环境变量和 .env 文件）::

    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...
    OPENAI_MODEL=...

每个测试都会发起真实的网络调用，因此会消耗 API token。当任一变量缺失时，
fixture 会给出清晰的跳过提示，确保默认运行行为永远不会被破坏。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.llm import LLMFactory, LLMInterface

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


@pytest.fixture
def live_llm() -> LLMInterface:
    model = os.environ.get("OPENAI_MODEL")
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    missing = [
        name
        for name, value in (
            ("OPENAI_MODEL", model),
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        pytest.skip(f"live tests skipped -- missing env vars: {missing}")

    assert model is not None and base_url is not None and api_key is not None
    return LLMFactory.create(
        "openai_compatible",
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_streaming_yields_non_empty_reply_from_real_api(
    live_llm: LLMInterface,
) -> None:
    messages = [{"role": "user", "content": "用一句话（不超过20字）中文回答：水的沸点是多少？"}]
    chunks = [
        c async for c in live_llm.chat_completion_stream(messages, system="你是一个简洁准确的助手")
    ]
    assert chunks, "expected at least one chunk from the streaming response"
    reply = "".join(chunks)
    assert reply.strip(), f"empty concatenated reply: {reply!r}"


@pytest.mark.live
@pytest.mark.asyncio
async def test_default_non_streaming_collects_full_reply_from_real_api(
    live_llm: LLMInterface,
) -> None:
    messages = [{"role": "user", "content": "用一句话（不超过20字）中文回答：地球距太阳多远？"}]
    reply = await live_llm.chat_completion(messages, system="你是一个简洁准确的助手")
    assert isinstance(reply, str)
    assert reply.strip(), f"empty reply: {reply!r}"
