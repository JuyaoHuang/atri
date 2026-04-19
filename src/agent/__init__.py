"""Agent package -- chat entry points.

Phase 4 exports: :class:`ChatAgent` (streaming + collect dual interface),
:class:`Persona` (frozen dataclass), and :func:`load_persona` (markdown +
frontmatter loader). The :class:`src.service_context.ServiceContext`
(US-AGT-005) is the intended top-level factory for ChatAgent instances.

Agent 包——聊天入口。

Phase 4 导出：:class:`ChatAgent`（流式 + 收集双接口）、:class:`Persona`
（冻结数据类）和 :func:`load_persona`（markdown + frontmatter 加载器）。
:class:`src.service_context.ServiceContext`（US-AGT-005）是 ChatAgent 实例
的顶层工厂。
"""

from src.agent.chat_agent import ChatAgent
from src.agent.persona import Persona, load_persona

__all__ = ["ChatAgent", "Persona", "load_persona"]
