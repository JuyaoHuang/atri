"""Tests for src/memory/long_term.py -- mem0 dual-mode backend.

Covers PRD US-MEM-006 acceptance criteria:
  (a) sdk mode constructs MemoryClient with api_key
  (b) local_deploy mode calls Memory.from_config with translated config
  (c) unresolved ${MEM0_API_KEY} placeholder raises ValueError
  (d) add delegates to underlying .add
  (e) search delegates and returns the results list

All mem0 backends are mocked -- no real network calls, no local qdrant/ollama
dependencies.

针对 src/memory/long_term.py 的测试——mem0 的双模后端。

覆盖 PRD US-MEM-006 的验收标准：
  (a) sdk 模式使用 api_key 构造 MemoryClient
  (b) local_deploy 模式调用 Memory.from_config，传入翻译后的配置
  (c) 未解析的 ${MEM0_API_KEY} 占位符抛出 ValueError
  (d) add 委托到底层 .add
  (e) search 委托并返回结果列表

所有 mem0 后端均被 mock——不会发起真实网络调用，也不依赖本地
qdrant/ollama。额外覆盖：未知 mode 报错、空/None api_key 报错、add 时
human/ai 角色被翻译为 user/assistant、search 的 bare list 响应兼容、
阈值过滤（§8.3 默认 0.3）、limit 生效、后端异常时 search 返回 []、
add 吞掉后端异常不破坏短期路径、api 后端的 embedder/llm 翻译、
graph_store 启用翻译，以及 close() 对 SaaS 模式的安全空操作。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.memory.long_term import LongTermMemory, _translate_local_deploy

# ---------------------------------------------------------------------------
# Config fixtures
# 配置固件
# ---------------------------------------------------------------------------


def _sdk_config(api_key: str = "m0-test-key") -> dict[str, Any]:
    return {"mode": "sdk", "sdk": {"api_key": api_key}}


def _local_deploy_config() -> dict[str, Any]:
    return {
        "mode": "local_deploy",
        "local_deploy": {
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": "./data/qdrant"},
            },
            "embedder": {
                "backend": "ollama",
                "ollama": {"model": "bge-m3", "base_url": "http://localhost:11434"},
            },
            "llm": {
                "backend": "ollama",
                "ollama": {"model": "qwen2.5:7b", "base_url": "http://localhost:11434"},
            },
            "graph_store": {"enabled": False},
        },
    }


# ---------------------------------------------------------------------------
# Constructor / mode validation
# 构造函数 / 模式校验
# ---------------------------------------------------------------------------


def test_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mem0.mode"):
        LongTermMemory({"mode": "invalid"})


def test_sdk_mode_constructs_memory_client() -> None:
    """PRD (a): sdk mode wires mem0.MemoryClient with the supplied api_key.

    PRD (a)：sdk 模式使用传入的 api_key 连接 mem0.MemoryClient。
    """
    with patch("mem0.MemoryClient") as mock_client_cls:
        ltm = LongTermMemory(_sdk_config(api_key="m0-abc123"))
        mock_client_cls.assert_called_once_with(api_key="m0-abc123")
        assert ltm.mode == "sdk"


def test_local_deploy_mode_calls_from_config() -> None:
    """PRD (b): local_deploy mode calls Memory.from_config with translated dict.

    PRD (b)：local_deploy 模式使用翻译后的字典调用 Memory.from_config。
    """
    with patch("mem0.Memory") as mock_memory_cls:
        ltm = LongTermMemory(_local_deploy_config())
        mock_memory_cls.from_config.assert_called_once()
        translated = mock_memory_cls.from_config.call_args.args[0]
        assert translated["vector_store"]["provider"] == "qdrant"
        assert translated["embedder"]["provider"] == "ollama"
        assert translated["embedder"]["config"]["model"] == "bge-m3"
        assert translated["embedder"]["config"]["ollama_base_url"] == ("http://localhost:11434")
        assert translated["llm"]["provider"] == "ollama"
        # graph_store disabled -> omitted entirely from translated dict.
        # graph_store 未启用 -> 翻译后的字典中完全不包含。
        assert "graph_store" not in translated
        assert ltm.mode == "local_deploy"


def test_unresolved_api_key_placeholder_raises() -> None:
    """PRD (c): '${MEM0_API_KEY}' literal surviving dotenv = misconfig.

    PRD (c)：'${MEM0_API_KEY}' 字面残留说明 dotenv 未解析 = 配置错误。
    """
    with pytest.raises(ValueError, match="unresolved"):
        LongTermMemory(_sdk_config(api_key="${MEM0_API_KEY}"))


def test_missing_api_key_raises() -> None:
    """Empty/None api_key also fails loudly rather than silently.

    空 / None 的 api_key 也应高声报错而不是静默放行。
    """
    with pytest.raises(ValueError):
        LongTermMemory({"mode": "sdk", "sdk": {}})


# ---------------------------------------------------------------------------
# Async delegation (add / search)
# 异步委托（add / search）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_delegates_to_underlying_backend() -> None:
    """PRD (d): add() forwards messages + ids to the backend.

    Roles are translated human->user / ai->assistant at the boundary so mem0's
    payload validator (which expects OpenAI-style roles) doesn't reject the call.

    PRD (d)：add() 将消息 + ids 转发给后端。

    在边界处角色被翻译（human->user / ai->assistant），以免 mem0 的载荷
    校验器（期望 OpenAI 风格角色）拒绝调用。
    """
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.add = MagicMock(return_value={"results": []})
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        messages = [
            {"role": "human", "content": "hello"},
            {"role": "ai", "content": "hi back"},
        ]
        await ltm.add(messages, user_id="alice", agent_id="atri", run_id="sess-1")

        mock_backend.add.assert_called_once()
        args, kwargs = mock_backend.add.call_args
        # The positional payload is the translated list (human -> user, ai -> assistant).
        # 位置载荷是被翻译后的列表（human -> user，ai -> assistant）。
        assert args[0] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi back"},
        ]
        assert kwargs == {"user_id": "alice", "agent_id": "atri", "run_id": "sess-1"}


@pytest.mark.asyncio
async def test_search_delegates_and_returns_results() -> None:
    """PRD (e): search() delegates and returns the results list.

    Modern mem0 clients reject top-level ``user_id`` / ``agent_id`` on search;
    the wrapper funnels them into ``filters={...}`` + ``top_k=limit``.

    PRD (e)：search() 委托并返回结果列表。

    新版 mem0 客户端拒绝 search 的顶层 ``user_id`` / ``agent_id`` 参数；
    包装层将其汇集到 ``filters={...}`` + ``top_k=limit`` 中。
    """
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(
            return_value={"results": [{"memory": "fact1", "score": 0.9}]}
        )
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        results = await ltm.search("drink preferences", user_id="alice", agent_id="atri")

        mock_backend.search.assert_called_once()
        args, kwargs = mock_backend.search.call_args
        assert args[0] == "drink preferences"
        assert kwargs["filters"] == {"user_id": "alice", "agent_id": "atri"}
        # 默认 limit
        assert kwargs["top_k"] == 5  # default limit
        assert results == [{"memory": "fact1", "score": 0.9}]


@pytest.mark.asyncio
async def test_search_accepts_bare_list_response() -> None:
    """mem0 versions/clients sometimes return a bare list instead of {'results': ...}.

    部分 mem0 版本 / 客户端有时返回裸列表而非 {'results': ...}。
    """
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(return_value=[{"memory": "fact-bare", "score": 0.8}])
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        results = await ltm.search("q", user_id="a", agent_id="b")
        assert results == [{"memory": "fact-bare", "score": 0.8}]


@pytest.mark.asyncio
async def test_search_filters_below_threshold() -> None:
    """Threshold filter is applied inside the wrapper (§8.3 default 0.3).

    阈值过滤在包装层内部应用（§8.3 默认 0.3）。
    """
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(
            return_value={
                "results": [
                    {"memory": "strong", "score": 0.9},
                    {"memory": "weak", "score": 0.1},
                ]
            }
        )
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        results = await ltm.search("q", user_id="a", agent_id="b", threshold=0.3)
        assert len(results) == 1
        assert results[0]["memory"] == "strong"


@pytest.mark.asyncio
async def test_search_respects_limit() -> None:
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(
            return_value={"results": [{"memory": f"fact{i}", "score": 0.9} for i in range(10)]}
        )
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        results = await ltm.search("q", user_id="a", agent_id="b", limit=3)
        assert len(results) == 3


@pytest.mark.asyncio
async def test_search_returns_empty_on_backend_error() -> None:
    """A broken mem0 must degrade gracefully -- we don't crash the chat turn.

    损坏的 mem0 必须优雅降级——不会导致聊天回合崩溃。
    """
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(side_effect=RuntimeError("network down"))
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        assert await ltm.search("q", user_id="a", agent_id="b") == []


@pytest.mark.asyncio
async def test_add_swallows_backend_errors() -> None:
    """mem0 failures must not break the short-term path (logged WARNING instead).

    mem0 失败不得中断短期路径（改为记录 WARNING）。
    """
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.add = MagicMock(side_effect=RuntimeError("boom"))
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        # Must not raise -- caller's on_round_complete keeps going.
        # 不得抛异常——调用方的 on_round_complete 可以继续执行。
        await ltm.add([], user_id="a", agent_id="b", run_id="c")


# ---------------------------------------------------------------------------
# Config translator helpers
# 配置翻译器辅助函数
# ---------------------------------------------------------------------------


def test_translate_local_deploy_with_api_backend() -> None:
    cfg = {
        "vector_store": {"provider": "qdrant", "config": {}},
        "embedder": {
            "backend": "api",
            "api": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-xxx",
                "base_url": "https://api.openai.com/v1",
            },
        },
        "llm": {
            "backend": "api",
            "api": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-yyy",
                "base_url": "https://api.openai.com/v1",
            },
        },
    }
    translated = _translate_local_deploy(cfg)
    assert translated["embedder"]["provider"] == "openai"
    assert translated["embedder"]["config"]["model"] == "text-embedding-3-small"
    assert translated["embedder"]["config"]["api_key"] == "sk-xxx"
    assert translated["embedder"]["config"]["openai_base_url"] == ("https://api.openai.com/v1")
    assert translated["llm"]["provider"] == "openai"
    assert translated["llm"]["config"]["model"] == "gpt-4o-mini"


def test_translate_local_deploy_enables_graph_when_flag_set() -> None:
    cfg = {
        "graph_store": {
            "enabled": True,
            "provider": "neo4j",
            "config": {"url": "bolt://localhost:7687", "username": "neo4j"},
        },
    }
    translated = _translate_local_deploy(cfg)
    assert translated["graph_store"]["provider"] == "neo4j"
    assert translated["graph_store"]["config"]["url"] == "bolt://localhost:7687"


def test_translate_llm_forwards_sampling_fields() -> None:
    """LLM blocks forward temperature / max_tokens / top_p to mem0 config."""
    cfg = {
        "llm": {
            "backend": "api",
            "api": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-x",
                "base_url": "https://api.openai.com/v1",
                "temperature": 0.25,
                "max_tokens": 512,
                "top_p": 0.9,
            },
        },
    }
    translated = _translate_local_deploy(cfg)
    llm_config = translated["llm"]["config"]
    assert llm_config["temperature"] == 0.25
    assert llm_config["max_tokens"] == 512
    assert llm_config["top_p"] == 0.9


def test_translate_llm_forwards_ollama_sampling_fields() -> None:
    """Ollama backend also forwards temperature when is_llm is True."""
    cfg = {
        "llm": {
            "backend": "ollama",
            "ollama": {
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.3,
            },
        },
    }
    translated = _translate_local_deploy(cfg)
    assert translated["llm"]["provider"] == "ollama"
    assert translated["llm"]["config"]["temperature"] == 0.3


def test_translate_embedder_drops_sampling_fields() -> None:
    """Embedder path ignores temperature/max_tokens/top_p even if yaml has them."""
    cfg = {
        "embedder": {
            "backend": "api",
            "api": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-e",
                "temperature": 0.5,  # should be dropped
                "max_tokens": 99,  # should be dropped
            },
        },
    }
    translated = _translate_local_deploy(cfg)
    emb_config = translated["embedder"]["config"]
    assert "temperature" not in emb_config
    assert "max_tokens" not in emb_config
    assert "top_p" not in emb_config


# ---------------------------------------------------------------------------
# close()
# close() 方法
# ---------------------------------------------------------------------------


def test_close_is_safe_noop_for_sdk_mode() -> None:
    with patch("mem0.MemoryClient") as mock_client_cls:
        # 无 .vector_store 属性
        mock_client_cls.return_value = MagicMock(spec=[])  # no .vector_store attr
        ltm = LongTermMemory(_sdk_config())
        # Must not raise even though SaaS backend has no vector_store.
        # 即便 SaaS 后端没有 vector_store 也不得抛异常。
        ltm.close()
