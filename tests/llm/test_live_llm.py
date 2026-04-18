"""Live end-to-end tests against a real LLM API.

These tests are gated by ``@pytest.mark.live`` and excluded from the
default ``pytest`` run. To execute them::

    uv run pytest tests/llm/test_live_llm.py -m live -v

Required: a project-root ``.env`` (git-ignored) containing::

    api_key=...
    base_url=...
    mddel_name=...     # intentional typo matches the provided .env

Each test makes a real network call and therefore costs API tokens. The
fixture skips with a clear message when the ``.env`` is missing or
incomplete so default-run behavior is never broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm import LLMFactory, LLMInterface

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip()
    return env


@pytest.fixture
def live_llm() -> LLMInterface:
    env = _load_env_file(_ENV_FILE)
    if not env:
        pytest.skip(f".env not found at {_ENV_FILE} -- live tests skipped")

    model = env.get("mddel_name") or env.get("model_name") or env.get("model")
    base_url = env.get("base_url")
    api_key = env.get("api_key")
    missing = [
        name
        for name, value in (
            ("model/mddel_name", model),
            ("base_url", base_url),
            ("api_key", api_key),
        )
        if not value
    ]
    if missing:
        pytest.skip(f".env missing required fields: {missing}")

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
        c
        async for c in live_llm.chat_completion_stream(
            messages, system="你是一个简洁准确的助手"
        )
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
    reply = await live_llm.chat_completion(
        messages, system="你是一个简洁准确的助手"
    )
    assert isinstance(reply, str)
    assert reply.strip(), f"empty reply: {reply!r}"
