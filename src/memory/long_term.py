"""mem0 long-term memory integration -- dual-mode backend (sdk + local_deploy).

Wraps :class:`mem0.MemoryClient` (SaaS) and :class:`mem0.Memory` (self-host)
behind one async-friendly API so callers (``MemoryManager``) stay mode-
agnostic. Blocking mem0 calls are dispatched via :func:`asyncio.to_thread`
so the event loop stays responsive during LLM-backed fact extraction.

Reference: docs/记忆系统设计讨论.md §4.1–§4.5 and §8.1–§8.4.

Config translation (local_deploy mode):

    Our yaml shape                   mem0 shape
    ----------------------------------------------------------
    vector_store.{provider, config}  vector_store.{provider, config}
    embedder.{backend='ollama',      embedder.{provider='ollama',
              ollama.{model,                   config.{model, ollama_base_url}}
              base_url}}
    embedder.{backend='api',         embedder.{provider,
              api.{provider, model,            config.{model, api_key,
              api_key, base_url}}                      openai_base_url}}
    llm.{backend, ollama/api}        llm.{provider, config}
    graph_store.{enabled=True, ...}  graph_store.{provider, config}
    graph_store.{enabled=False}      (omitted)
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger


def _is_unresolved_placeholder(value: Any) -> bool:
    """Return True when ``value`` still looks like ``${VAR}`` post-dotenv load."""
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def _translate_local_deploy(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate our ``mem0.local_deploy`` yaml block into mem0's config shape.

    mem0's :func:`Memory.from_config` expects a flat dict whose top-level
    keys are ``vector_store`` / ``embedder`` / ``llm`` (plus optional
    ``graph_store``). Each is a ``{provider, config}`` pair. Our yaml groups
    embedder/llm under a ``backend`` switch (``ollama`` vs ``api``) with
    sub-dicts per backend -- this helper picks the active branch and maps
    field names (``base_url`` → ``ollama_base_url`` / ``openai_base_url``).
    """
    out: dict[str, Any] = {}

    vs = cfg.get("vector_store")
    if vs:
        # Already matches mem0's {provider, config} shape.
        out["vector_store"] = vs

    embedder_cfg = cfg.get("embedder") or {}
    if embedder_cfg:
        out["embedder"] = _translate_backend_block(embedder_cfg, is_llm=False)

    llm_cfg = cfg.get("llm") or {}
    if llm_cfg:
        out["llm"] = _translate_backend_block(llm_cfg, is_llm=True)

    graph_cfg = cfg.get("graph_store") or {}
    if graph_cfg.get("enabled"):
        out["graph_store"] = {
            "provider": graph_cfg.get("provider", "neo4j"),
            "config": graph_cfg.get("config", {}),
        }

    return out


def _translate_backend_block(block: dict[str, Any], *, is_llm: bool) -> dict[str, Any]:
    """Translate an embedder/llm block with a ``backend`` selector."""
    backend = block.get("backend", "ollama")
    if backend == "ollama":
        ol = block.get("ollama") or {}
        return {
            "provider": "ollama",
            "config": {
                "model": ol.get("model"),
                "ollama_base_url": ol.get("base_url"),
            },
        }
    api = block.get("api") or {}
    provider = api.get("provider", "openai")
    config: dict[str, Any] = {
        "model": api.get("model"),
        "api_key": api.get("api_key"),
    }
    # mem0's openai provider accepts ``openai_base_url`` at the config level.
    if api.get("base_url"):
        config["openai_base_url"] = api.get("base_url")
    return {"provider": provider, "config": config}


class LongTermMemory:
    """Unified async wrapper over mem0's SDK/local-deploy backends.

    Callers pass the ``mem0`` subtree of ``memory_config.yaml``; the class
    picks the right backend and exposes ``add`` / ``search`` / ``close``
    with a single shape.
    """

    def __init__(self, mem0_config: dict[str, Any]) -> None:
        mode = mem0_config.get("mode")
        if mode not in ("sdk", "local_deploy"):
            raise ValueError(f"mem0.mode must be 'sdk' or 'local_deploy', got {mode!r}")
        self.mode: str = mode
        self._backend: Any = None

        if mode == "sdk":
            sdk_cfg = mem0_config.get("sdk") or {}
            api_key = sdk_cfg.get("api_key")
            if not api_key or _is_unresolved_placeholder(api_key):
                raise ValueError(
                    f"mem0 sdk api_key is missing or unresolved: {api_key!r}. "
                    f"Set MEM0_API_KEY in .env so config loader can substitute it."
                )
            from mem0 import MemoryClient

            self._backend = MemoryClient(api_key=api_key)
        else:
            from mem0 import Memory

            translated = _translate_local_deploy(mem0_config.get("local_deploy") or {})
            self._backend = Memory.from_config(translated)

        logger.info(f"LongTermMemory ready | mode={mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        agent_id: str,
        run_id: str,
    ) -> None:
        """Persist a batch of messages to mem0 (runs in a worker thread).

        Errors are logged as WARNING and swallowed -- the short-term path
        continues uninterrupted. mem0 v3's ADD-only algorithm makes a
        dropped ``add()`` cheap to recover on the next window (and retries
        are idempotent if we ever add them).
        """
        try:
            await asyncio.to_thread(
                self._backend.add,
                messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"mem0.add failed | {exc!r}")

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 5,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Retrieve related memories for the current user/agent pair.

        Returns up to ``limit`` hits whose ``score`` is ``>= threshold``
        (defaults mirror §8.3). On any backend error returns ``[]`` and
        logs WARNING -- the LLM call still proceeds without long-term
        context rather than failing the whole turn.
        """
        try:
            raw = await asyncio.to_thread(
                self._backend.search,
                query,
                user_id=user_id,
                agent_id=agent_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"mem0.search failed | {exc!r}")
            return []

        # mem0 v1.1+ wraps results in {"results": [...]}; older paths and
        # the SaaS client sometimes return a bare list. Handle both.
        if isinstance(raw, dict):
            items: list[dict[str, Any]] = list(raw.get("results", []))
        elif isinstance(raw, list):
            items = list(raw)
        else:
            items = []

        filtered = [r for r in items if float(r.get("score", 1.0)) >= threshold]
        return filtered[:limit]

    def close(self) -> None:
        """Best-effort cleanup. No-op for SaaS; closes the Qdrant client
        handle for local_deploy so embedded rocksdb locks release promptly."""
        backend = getattr(self, "_backend", None)
        if backend is None:
            return
        try:
            vs = getattr(backend, "vector_store", None)
            client = getattr(vs, "client", None) if vs is not None else None
            if client is not None and hasattr(client, "close"):
                client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"LongTermMemory.close minor issue: {exc!r}")


__all__ = ["LongTermMemory"]
