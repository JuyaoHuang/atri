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

mem0 长期记忆集成——双模式后端（sdk + local_deploy）。

将 :class:`mem0.MemoryClient` (SaaS) 和 :class:`mem0.Memory` (自托管) 包装
在一个异步友好的统一 API 背后，使调用方（``MemoryManager``）对模式保持
无感。阻塞型的 mem0 调用通过 :func:`asyncio.to_thread` 派发，这样 LLM 驱动
的事实抽取过程中事件循环仍然可响应。

参考：docs/记忆系统设计讨论.md §4.1–§4.5 与 §8.1–§8.4。

配置转换（local_deploy 模式）：

    我们的 yaml 形态                 mem0 形态
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
    graph_store.{enabled=False}      （省略）
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger


def _is_unresolved_placeholder(value: Any) -> bool:
    """Return True when ``value`` still looks like ``${VAR}`` post-dotenv load.

    当 ``value`` 在 dotenv 加载后仍形如 ``${VAR}`` 时返回 True。
    """
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


# Our short-term layer tags messages with internal role names (``human`` /
# ``ai``) but mem0 -- like every other OpenAI-compatible LLM SDK -- expects
# ``user`` / ``assistant``. Feeding our raw roles to ``mem0.add`` trips the
# payload validator. Translate at the boundary so the rest of the codebase
# stays on the human/ai vocabulary.
# 我们的短期层用内部角色名（``human`` / ``ai``）标记消息，但 mem0 —— 和所有
# OpenAI 兼容的 LLM SDK 一样 —— 期望的是 ``user`` / ``assistant``。直接把原始
# 角色喂给 ``mem0.add`` 会触发载荷校验错误。因此在边界上做一次翻译，让代码库
# 其它位置继续使用 human/ai 词汇体系。
_MEM0_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
}


def _translate_messages_for_mem0(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map internal role labels to mem0/OpenAI roles; keep content verbatim.

    将内部角色标签映射为 mem0/OpenAI 的角色；内容保持原样不变。
    """
    translated: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        translated.append(
            {
                "role": _MEM0_ROLE_MAP.get(role, role),
                "content": msg.get("content", ""),
            }
        )
    return translated


def _translate_local_deploy(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate our ``mem0.local_deploy`` yaml block into mem0's config shape.

    mem0's :func:`Memory.from_config` expects a flat dict whose top-level
    keys are ``vector_store`` / ``embedder`` / ``llm`` (plus optional
    ``graph_store``). Each is a ``{provider, config}`` pair. Our yaml groups
    embedder/llm under a ``backend`` switch (``ollama`` vs ``api``) with
    sub-dicts per backend -- this helper picks the active branch and maps
    field names (``base_url`` → ``ollama_base_url`` / ``openai_base_url``).

    将我们的 ``mem0.local_deploy`` yaml 配置块翻译为 mem0 期望的配置形态。

    mem0 的 :func:`Memory.from_config` 期望一个扁平字典，其顶层键为
    ``vector_store`` / ``embedder`` / ``llm``（以及可选的 ``graph_store``），
    每个键对应一个 ``{provider, config}`` 对。我们的 yaml 则在 embedder/llm
    下按 ``backend`` 开关（``ollama`` 对比 ``api``）再分子字典——本辅助函数
    挑出激活的分支，并映射字段名（``base_url`` → ``ollama_base_url`` /
    ``openai_base_url``）。
    """
    out: dict[str, Any] = {}

    vs = cfg.get("vector_store")
    if vs:
        # Already matches mem0's {provider, config} shape.
        # 已匹配 mem0 的 {provider, config} 形态。
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
    """Translate an embedder/llm block with a ``backend`` selector.

    When ``is_llm`` is ``True`` the active backend's LLM-only sampling fields
    (``temperature`` / ``max_tokens`` / ``top_p``) are forwarded into the
    mem0 ``config`` dict. Embedder calls ignore those keys even if present
    in the yaml.

    翻译带 ``backend`` 选择器的 embedder/llm 配置块。

    当 ``is_llm`` 为 ``True`` 时，把当前后端专属的 LLM 采样字段
    （``temperature`` / ``max_tokens`` / ``top_p``）透传到 mem0 的 ``config``
    字典中。embedder 调用会忽略这些键（即便 yaml 里写了也一样）。
    """
    backend = block.get("backend", "ollama")
    if backend == "ollama":
        sub = block.get("ollama") or {}
        provider = "ollama"
        config: dict[str, Any] = {
            "model": sub.get("model"),
            "ollama_base_url": sub.get("base_url"),
        }
    else:
        sub = block.get("api") or {}
        provider = sub.get("provider", "openai")
        config = {
            "model": sub.get("model"),
            "api_key": sub.get("api_key"),
        }
        # mem0's openai provider accepts ``openai_base_url`` at the config level.
        # mem0 的 openai 提供商在 config 级别接受 ``openai_base_url``。
        if sub.get("base_url"):
            config["openai_base_url"] = sub.get("base_url")

    # LLM-only sampling knobs -- embedder path drops these silently so yaml
    # authors can keep a uniform backend block without surprising side effects.
    # LLM 专属采样参数——embedder 路径会静默丢弃，便于作者保持统一的后端
    # 块而不会产生意外副作用。
    if is_llm:
        for key in ("temperature", "max_tokens", "top_p"):
            if key in sub:
                config[key] = sub[key]

    return {"provider": provider, "config": config}


class LongTermMemory:
    """Unified async wrapper over mem0's SDK/local-deploy backends.

    Callers pass the ``mem0`` subtree of ``memory_config.yaml``; the class
    picks the right backend and exposes ``add`` / ``search`` / ``close``
    with a single shape.

    mem0 SDK/本地部署后端的统一异步包装。

    调用方传入 ``memory_config.yaml`` 中的 ``mem0`` 子树；类会挑选合适的
    后端，并以统一的形态对外暴露 ``add`` / ``search`` / ``close``。
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
    # 公共 API
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        agent_id: str,
        run_id: str,
    ) -> None:
        """Persist a batch of messages to mem0 (runs in a worker thread).

        Roles are translated at the boundary (``human`` -> ``user``,
        ``ai`` -> ``assistant``) to match mem0's payload validator. Errors
        are logged as WARNING and swallowed -- the short-term path continues
        uninterrupted. mem0 v3's ADD-only algorithm makes a dropped
        ``add()`` cheap to recover on the next window.

        将一批消息持久化到 mem0（在工作线程中运行）。

        角色在边界上被翻译（``human`` -> ``user``、``ai`` -> ``assistant``），
        以匹配 mem0 的载荷校验器。错误以 WARNING 级别记录并被吞掉——短期路径
        不受影响继续推进。mem0 v3 的纯 ADD 算法让一次失败的 ``add()`` 可以
        在下一个窗口低成本地恢复。
        """
        try:
            payload = _translate_messages_for_mem0(messages)
            await asyncio.to_thread(
                self._backend.add,
                payload,
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

        Modern mem0 clients reject top-level entity kwargs on ``search`` and
        require ``filters={...}`` plus ``top_k=...`` instead. Returns up to
        ``limit`` hits whose ``score`` is ``>= threshold`` (defaults mirror
        §8.3). On any backend error returns ``[]`` and logs WARNING so the
        LLM call proceeds without long-term context rather than failing the
        whole turn.

        为当前 user/agent 对检索相关记忆。

        新版 mem0 客户端不再接受顶层的实体 kwargs，而要求通过
        ``filters={...}`` 与 ``top_k=...`` 来传参。返回 ``score`` ``>=
        threshold`` 的最多 ``limit`` 条命中（默认值对齐 §8.3）。发生任何
        后端错误时返回 ``[]`` 并记录 WARNING，从而让 LLM 调用在不带长期
        上下文的情况下继续推进，而不是让整轮失败。
        """
        filters = {"user_id": user_id, "agent_id": agent_id}
        try:
            raw = await asyncio.to_thread(
                self._backend.search,
                query,
                filters=filters,
                top_k=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"mem0.search failed | {exc!r}")
            return []

        # mem0 v1.1+ wraps results in {"results": [...]}; older paths and
        # the SaaS client sometimes return a bare list. Handle both.
        # mem0 v1.1+ 将结果包裹在 {"results": [...]} 中；早期路径和 SaaS
        # 客户端有时返回裸列表。两种形态都需要处理。
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
        handle for local_deploy so embedded rocksdb locks release promptly.

        尽力而为的清理。SaaS 模式下为空操作；local_deploy 模式下关闭 Qdrant
        客户端句柄，让内嵌 rocksdb 的锁尽快释放。
        """
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
