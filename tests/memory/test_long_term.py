"""Tests for src/memory/long_term.py -- mem0 dual-mode backend.

Covers PRD US-MEM-006 acceptance criteria:
  (a) sdk mode constructs MemoryClient with api_key
  (b) local_deploy mode calls Memory.from_config with translated config
  (c) unresolved ${MEM0_API_KEY} placeholder raises ValueError
  (d) add delegates to underlying .add
  (e) search delegates and returns the results list

All mem0 backends are mocked -- no real network calls, no local qdrant/ollama
dependencies.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.memory.long_term import LongTermMemory, _translate_local_deploy

# ---------------------------------------------------------------------------
# Config fixtures
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
# ---------------------------------------------------------------------------


def test_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mem0.mode"):
        LongTermMemory({"mode": "invalid"})


def test_sdk_mode_constructs_memory_client() -> None:
    """PRD (a): sdk mode wires mem0.MemoryClient with the supplied api_key."""
    with patch("mem0.MemoryClient") as mock_client_cls:
        ltm = LongTermMemory(_sdk_config(api_key="m0-abc123"))
        mock_client_cls.assert_called_once_with(api_key="m0-abc123")
        assert ltm.mode == "sdk"


def test_local_deploy_mode_calls_from_config() -> None:
    """PRD (b): local_deploy mode calls Memory.from_config with translated dict."""
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
        assert "graph_store" not in translated
        assert ltm.mode == "local_deploy"


def test_unresolved_api_key_placeholder_raises() -> None:
    """PRD (c): '${MEM0_API_KEY}' literal surviving dotenv = misconfig."""
    with pytest.raises(ValueError, match="unresolved"):
        LongTermMemory(_sdk_config(api_key="${MEM0_API_KEY}"))


def test_missing_api_key_raises() -> None:
    """Empty/None api_key also fails loudly rather than silently."""
    with pytest.raises(ValueError):
        LongTermMemory({"mode": "sdk", "sdk": {}})


# ---------------------------------------------------------------------------
# Async delegation (add / search)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_delegates_to_underlying_backend() -> None:
    """PRD (d): add() forwards messages + ids to the backend."""
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.add = MagicMock(return_value={"results": []})
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        messages = [{"role": "human", "content": "hello"}]
        await ltm.add(messages, user_id="alice", agent_id="atri", run_id="sess-1")

        mock_backend.add.assert_called_once_with(
            messages, user_id="alice", agent_id="atri", run_id="sess-1"
        )


@pytest.mark.asyncio
async def test_search_delegates_and_returns_results() -> None:
    """PRD (e): search() delegates and returns the results list."""
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(
            return_value={"results": [{"memory": "fact1", "score": 0.9}]}
        )
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        results = await ltm.search("drink preferences", user_id="alice", agent_id="atri")

        mock_backend.search.assert_called_once()
        # Call signature: (query, user_id=..., agent_id=...)
        args, kwargs = mock_backend.search.call_args
        assert args[0] == "drink preferences"
        assert kwargs["user_id"] == "alice"
        assert kwargs["agent_id"] == "atri"
        assert results == [{"memory": "fact1", "score": 0.9}]


@pytest.mark.asyncio
async def test_search_accepts_bare_list_response() -> None:
    """mem0 versions/clients sometimes return a bare list instead of {'results': ...}."""
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(return_value=[{"memory": "fact-bare", "score": 0.8}])
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        results = await ltm.search("q", user_id="a", agent_id="b")
        assert results == [{"memory": "fact-bare", "score": 0.8}]


@pytest.mark.asyncio
async def test_search_filters_below_threshold() -> None:
    """Threshold filter is applied inside the wrapper (§8.3 default 0.3)."""
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
    """A broken mem0 must degrade gracefully -- we don't crash the chat turn."""
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.search = MagicMock(side_effect=RuntimeError("network down"))
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        assert await ltm.search("q", user_id="a", agent_id="b") == []


@pytest.mark.asyncio
async def test_add_swallows_backend_errors() -> None:
    """mem0 failures must not break the short-term path (logged WARNING instead)."""
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_backend = MagicMock()
        mock_backend.add = MagicMock(side_effect=RuntimeError("boom"))
        mock_client_cls.return_value = mock_backend

        ltm = LongTermMemory(_sdk_config())
        # Must not raise -- caller's on_round_complete keeps going.
        await ltm.add([], user_id="a", agent_id="b", run_id="c")


# ---------------------------------------------------------------------------
# Config translator helpers
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


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_is_safe_noop_for_sdk_mode() -> None:
    with patch("mem0.MemoryClient") as mock_client_cls:
        mock_client_cls.return_value = MagicMock(spec=[])  # no .vector_store attr
        ltm = LongTermMemory(_sdk_config())
        # Must not raise even though SaaS backend has no vector_store.
        ltm.close()
