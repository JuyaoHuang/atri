"""Live LLM smoke test -- exercises OpenAICompatibleLLM against a real API.

Reads credentials from atri/.env (git-ignored) and runs two rounds:
  1. Streaming: tokens printed as they arrive
  2. Non-streaming (default impl): full reply collected into a string

Run:
    uv run python scripts/live_llm_smoke.py

The .env file is expected to contain at least::

    api_key=...
    base_url=...
    mddel_name=...       # yes, the typo is intentional to match the provided .env
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Force UTF-8 stdout so CJK streams render correctly on Windows consoles
# that default to GBK. Silent fallback when the stream is not a TextIOWrapper.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from loguru import logger  # noqa: E402

from src.llm import LLMFactory  # noqa: E402 -- after sys.path setup
from src.utils.logger import init_logger  # noqa: E402


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


async def _run_streaming(llm) -> None:
    logger.info("--- streaming ---")
    messages = [{"role": "user", "content": "用一句话（不超过30字）中文回答：水的沸点是多少？"}]
    collected: list[str] = []
    async for chunk in llm.chat_completion_stream(messages, system="你是一个简洁准确的助手"):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        collected.append(chunk)
    sys.stdout.write("\n")
    logger.info(
        "streaming finished | chunks={} total_chars={}",
        len(collected),
        sum(len(c) for c in collected),
    )


async def _run_non_streaming(llm) -> None:
    logger.info("--- non-streaming (default impl collects stream) ---")
    messages = [{"role": "user", "content": "用一句话（不超过30字）中文回答：地球距太阳多远？"}]
    text = await llm.chat_completion(messages, system="你是一个简洁准确的助手")
    logger.info("non-streaming finished | length={}", len(text))
    print(text)


async def main() -> None:
    init_logger()
    env = _load_env_file(_REPO_ROOT / ".env")
    if not env:
        logger.error(".env not found or empty at {}", _REPO_ROOT / ".env")
        sys.exit(1)

    model = env.get("mddel_name") or env.get("model_name") or env.get("model")
    base_url = env.get("base_url")
    api_key = env.get("api_key")
    if not (model and base_url and api_key):
        logger.error(
            ".env missing required fields | model={} base_url={} api_key_set={}",
            bool(model),
            bool(base_url),
            bool(api_key),
        )
        sys.exit(1)

    logger.info("creating LLM | model={} base_url={}", model, base_url)
    llm = LLMFactory.create(
        "openai_compatible",
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
    )

    await _run_streaming(llm)
    await _run_non_streaming(llm)
    logger.info("live smoke test PASSED")


if __name__ == "__main__":
    asyncio.run(main())
