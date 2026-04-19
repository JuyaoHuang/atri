"""ChatAgent -- Phase 4 composition layer.

Combines :class:`LLMInterface` + :class:`MemoryManager` + :class:`Persona`
into the persona-driven chat entry point. Exposes a streaming :meth:`chat`
and a non-streaming :meth:`chat_collect` (mirroring :class:`LLMInterface`'s
dual-interface pattern) and automatically commits the round to memory after
the stream exhausts.

Key design points (per docs/Phase4_执行规格.md §US-AGT-003 and strategy
decisions S1a / S1b / S5):

* ``user_input`` is passed **raw** (un-cleaned) to both
  :meth:`MemoryManager.build_llm_context` (payload position [6] / long-term
  search) and :meth:`MemoryManager.on_round_complete` (L1 is applied inside
  the manager). ChatAgent does no cleaning itself.
* ``persona.system_prompt`` is forwarded to ``build_llm_context`` as payload
  position [1]; :meth:`LLMInterface.chat_completion_stream` is called without
  a separate ``system=`` kwarg to avoid double-prepend.
* The error path (``LLMError`` from the stream) is added in US-AGT-004;
  this module ships only the success path so each story is independently
  verifiable.

Reference: docs/Phase4_执行规格.md §US-AGT-003, docs/项目架构设计.md §2.5,
docs/记忆系统设计讨论.md §6.1 (revised data flow),
docs/LLM调用层设计讨论.md §2.2.

ChatAgent——Phase 4 组合层。

将 :class:`LLMInterface` + :class:`MemoryManager` + :class:`Persona` 组合为
由 persona 驱动的聊天入口。暴露流式的 :meth:`chat` 与非流式的
:meth:`chat_collect`（镜像 :class:`LLMInterface` 的双接口模式），并在流结
束后自动将本轮提交给记忆层。

关键设计点（对齐 docs/Phase4_执行规格.md §US-AGT-003 以及策略决策
S1a / S1b / S5）：

* ``user_input`` 以**原样（未清洗）**同时传给
  :meth:`MemoryManager.build_llm_context`（载荷位置 [6] / 长期检索）
  与 :meth:`MemoryManager.on_round_complete`（L1 在 manager 内部执行）。
  ChatAgent 自身不做任何清洗。
* ``persona.system_prompt`` 作为载荷位置 [1] 通过 ``build_llm_context``
  注入；调用 :meth:`LLMInterface.chat_completion_stream` 时**不**额外传
  ``system=`` 关键字，避免重复前置。
* 错误路径（流中抛出 ``LLMError``）由 US-AGT-004 增补；本模块只承载
  成功路径，让每个 US 能独立验证。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from loguru import logger

from src.agent.persona import Persona
from src.llm.exceptions import LLMError
from src.llm.interface import LLMInterface
from src.memory.manager import MemoryManager

# Threshold below which a successful (non-errored) LLM reply is deemed
# "suspiciously short" and logged as WARNING. Chosen empirically: the atri
# persona's shortest well-formed replies are ~30+ chars (opening with
# "（动作）" bracket + body). A reply stripped to <10 chars is almost
# certainly an upstream truncation (observed cases: 1 single "（"). The
# log is purely observational -- it does NOT change commit / trigger
# behavior. Future iterations may promote this to a retry / error path
# based on real-world frequency data gathered via these WARNINGs.
# 成功路径下 LLM 回复被视为"可疑过短"并打 WARNING 的阈值。经验选取：atri
# persona 最短的完整回复约 30+ 字符（以"（动作）"括号开头 + 正文）。
# 剥离空白后 <10 字符几乎必然是上游截断（实测案例：单个 "（"）。日志纯
# 观察性——**不改变**提交 / 触发行为。未来可依据真实频率数据再决定是否
# 升级为重试 / 错误路径。
_SUSPICIOUS_REPLY_MIN_CHARS = 10


class ChatAgent:
    """Persona-driven streaming chat entry point.

    Composition layer that owns no state of its own beyond the three
    injected dependencies. One ChatAgent per ``(character_id, user_id)``
    pair -- the :class:`src.service_context.ServiceContext` (US-AGT-005)
    owns the lifecycle.

    由 persona 驱动的流式聊天入口。

    组合层，除三项注入依赖外不持有任何自有状态。每个
    ``(character_id, user_id)`` 对对应一个 ChatAgent——生命周期由
    :class:`src.service_context.ServiceContext`（US-AGT-005）管理。
    """

    def __init__(
        self,
        llm: LLMInterface,
        memory_manager: MemoryManager,
        persona: Persona,
    ) -> None:
        self.llm = llm
        self.memory_manager = memory_manager
        self.persona = persona

    async def chat(self, user_input: str) -> AsyncIterator[str]:
        """Stream LLM tokens for ``user_input`` and auto-commit the round.

        Sequence (success path, per §6.1):
          1. ``messages = await memory_manager.build_llm_context(
             user_input, system_prompt=persona.system_prompt)`` -- assembles
             the 6-segment payload; ``user_input`` lands raw at position [6]
             and also drives the long-term facts search at position [2].
          2. ``async for chunk in llm.chat_completion_stream(messages):``
             yield each chunk to the caller while accumulating the full
             reply.
          3. After the stream exhausts, call
             ``await memory_manager.on_round_complete(user_msg, ai_msg)``
             exactly once with user_msg = ``{role:'human', content: user_input}``
             (raw) and ai_msg = ``{role:'ai', content: reply, name:
             persona.name}``. MemoryManager applies L1 to the user message
             internally, appends to chat_history / recent_messages, and
             may fire L3 / L4 triggers.

        Error path (US-AGT-004): if the stream raises ``LLMError``, yield
        an error sentinel to the caller, call
        :meth:`MemoryManager.append_system_note`, and skip
        ``on_round_complete`` so the failed turn does not count as a round.

        Args:
            user_input: The raw user message (text-typed or ASR transcript).
                Not cleaned before ``build_llm_context`` / ``on_round_complete``;
                L1 cleansing happens inside MemoryManager.

        Yields:
            str: Each chunk from the underlying LLM stream in arrival order.

        为 ``user_input`` 流式拉取 LLM token 并在结束后自动提交本轮。

        顺序（成功路径，对齐 §6.1）：
          1. ``messages = await memory_manager.build_llm_context(
             user_input, system_prompt=persona.system_prompt)``——组装
             6 段载荷；``user_input`` 以原样落在位置 [6]，同时驱动位置 [2]
             的长期事实检索。
          2. ``async for chunk in llm.chat_completion_stream(messages):``
             将每个 chunk yield 给调用方，同时累积完整回复。
          3. 流结束后调用一次
             ``await memory_manager.on_round_complete(user_msg, ai_msg)``，
             其中 user_msg = ``{role:'human', content: user_input}``（原样），
             ai_msg = ``{role:'ai', content: reply, name: persona.name}``。
             MemoryManager 在内部对用户消息执行 L1，追加到 chat_history /
             recent_messages，并可能触发 L3 / L4。

        错误路径（US-AGT-004）：若流抛出 ``LLMError``，向调用方 yield
        错误哨兵，调用 :meth:`MemoryManager.append_system_note`，并跳过
        ``on_round_complete``，使失败的一轮不计数。

        参数：
            user_input：原始用户消息（文字输入或 ASR 转写）。传给
                ``build_llm_context`` / ``on_round_complete`` 之前**不**做清洗，
                L1 清洗由 MemoryManager 内部完成。

        产出：
            str：底层 LLM 流的每个 chunk（按到达顺序）。
        """
        messages = await self.memory_manager.build_llm_context(
            user_input,
            system_prompt=self.persona.system_prompt,
        )

        reply_chunks: list[str] = []
        try:
            async for chunk in self.llm.chat_completion_stream(messages):
                reply_chunks.append(chunk)
                yield chunk
        except LLMError as exc:
            # Error path (S4): surface the failure to the caller as a final
            # sentinel chunk, persist it as a chat_history system row, and
            # bail out WITHOUT counting the round. `append_system_note` does
            # not touch total_rounds / recent_messages / triggers (see
            # MemoryManager.append_system_note invariants).
            # 错误路径（S4）：将失败作为末尾的哨兵 chunk 告知调用方，持久化
            # 为 chat_history 的 system 行，并在**不**计入轮次的情况下退出。
            # `append_system_note` 不触碰 total_rounds / recent_messages /
            # 触发器（见 MemoryManager.append_system_note 的不变式）。
            error_text = f"[LLM call failed: {type(exc).__name__}: {exc}]"
            yield error_text
            self.memory_manager.append_system_note(error_text)
            return

        reply = "".join(reply_chunks)

        # Observational WARNING when a successful (non-errored) stream yields
        # a suspiciously short reply — almost always an upstream truncation
        # (e.g., DeepSeek/SiliconFlow occasionally closes stream after 1 token
        # with finish_reason=length/content_filter). Does NOT change behavior:
        # the round is still committed via on_round_complete so chat_history /
        # recent_messages stay honest. Frontend / TTS may see a single "（".
        # 成功（非错误）路径下 stream 产出"可疑过短"回复时打 WARNING——几乎
        # 总是上游截断（如 DeepSeek/SiliconFlow 偶发在 1 token 后关流，
        # finish_reason=length/content_filter）。**不改变**行为：仍会通过
        # on_round_complete 提交，使 chat_history / recent_messages 忠实记录。
        # 前端 / TTS 可能看到单个 "（"。
        if len(reply.strip()) < _SUSPICIOUS_REPLY_MIN_CHARS:
            logger.warning(
                "ChatAgent suspiciously short LLM reply | character={} | "
                "user_id={} | len={} | reply={!r} | possible upstream truncation",
                self.persona.character_id,
                self.memory_manager.user_id,
                len(reply),
                reply,
            )

        await self.memory_manager.on_round_complete(
            {"role": "human", "content": user_input},
            {"role": "ai", "content": reply, "name": self.persona.name},
        )

    async def chat_collect(self, user_input: str) -> str:
        """Collect :meth:`chat`'s streaming output into one string.

        Default implementation iterates :meth:`chat` so ``on_round_complete``
        still fires exactly once. No separate LLM call path -- subclasses
        may override when a non-streaming API is meaningfully cheaper.

        将 :meth:`chat` 的流式输出收集为单个字符串。

        默认实现通过迭代 :meth:`chat` 实现，使 ``on_round_complete`` 仍然
        恰好触发一次。不走独立的 LLM 调用路径——当非流式 API 显著更廉价
        时，子类可以覆盖此方法。
        """
        chunks = [chunk async for chunk in self.chat(user_input)]
        return "".join(chunks)


__all__ = ["ChatAgent"]
