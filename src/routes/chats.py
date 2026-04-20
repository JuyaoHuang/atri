"""Chat session CRUD REST API routes.
聊天会话 CRUD REST API 路由。

Provides endpoints for chat session management:
提供聊天会话管理端点：

- List chats (with optional character filter)
- Create chat with LLM-generated title
- Get chat details with message history
- Update chat title
- Delete chat

Reference: docs/Phase5_执行规格.md §US-SRV-005
"""

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel

from src.llm.exceptions import LLMError
from src.llm.factory import create_from_role

router = APIRouter(prefix="/api/chats", tags=["chats"])


class CreateChatRequest(BaseModel):
    """Request body for creating a new chat. / 创建新聊天的请求体。"""

    character_id: str
    first_message: str


class CreateChatResponse(BaseModel):
    """Response for chat creation. / 聊天创建响应。"""

    id: str
    title: str
    character_id: str
    created_at: str


class UpdateChatTitleRequest(BaseModel):
    """Request body for updating chat title. / 更新聊天标题的请求体。"""

    title: str


class ChatListItem(BaseModel):
    """Chat list item metadata. / 聊天列表项元数据。"""

    id: str
    title: str
    character_id: str
    created_at: str
    updated_at: str
    message_count: int


class ChatMessage(BaseModel):
    """Chat message. / 聊天消息。"""

    role: str
    content: str
    timestamp: str
    name: str | None = None


class ChatDetailResponse(BaseModel):
    """Chat detail with messages. / 聊天详情（含消息）。"""

    metadata: ChatListItem
    messages: list[ChatMessage]


async def _generate_title_with_llm(first_message: str, llm_config: dict) -> str | None:
    """Generate chat title using LLM.
    使用 LLM 生成聊天标题。

    Args:
        first_message: User's first message in the chat.
                       用户在聊天中的第一条消息。
        llm_config: LLM configuration dict.
                    LLM 配置字典。

    Returns:
        Generated title or None if LLM call fails.
        生成的标题，如果 LLM 调用失败则返回 None。
    """
    prompt = f"""请为以下用户消息生成一个简短的聊天标题（5-10个字）。
要求：
1. 概括对话的核心主题
2. 使用用户的语言
3. 不要使用引号或标点符号
4. 只返回标题文本，不要任何解释

用户消息：{first_message}

标题："""

    try:
        llm = create_from_role("title_gen", llm_config)
        messages = [{"role": "user", "content": prompt}]

        # Collect streaming response
        # 收集流式响应
        chunks = []
        async for chunk in llm.chat_completion_stream(messages):
            chunks.append(chunk)

        title = "".join(chunks).strip()

        # Validate title length (should be short)
        # 验证标题长度（应该简短）
        if len(title) > 50:
            logger.warning(f"Generated title too long ({len(title)} chars), truncating")
            title = title[:50]

        return title if title else None

    except LLMError as e:
        logger.warning(f"LLM title generation failed: {e}, will use fallback")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in title generation: {e}")
        return None


def _fallback_title(first_message: str) -> str:
    """Generate fallback title by truncating first message.
    通过截取首条消息生成降级标题。

    Args:
        first_message: User's first message.
                       用户的首条消息。

    Returns:
        Truncated title (max 20 chars).
        截取的标题（最多 20 字符）。
    """
    return first_message[:20] if len(first_message) > 20 else first_message


@router.get("", response_model=list[ChatListItem])
async def list_chats(
    request: Request,
    character_id: str | None = Query(None, description="Filter by character ID"),
) -> list[ChatListItem]:
    """List user's chat sessions, sorted by updated_at descending.
    列出用户的聊天会话，按 updated_at 降序排序。

    Args:
        character_id: Optional character filter.
                      可选的角色过滤器。

    Returns:
        List of chat metadata.
        聊天元数据列表。
    """
    storage = request.app.state.storage
    user_id = "default"  # Phase 5: hardcoded user_id

    chats = await storage.list_chats(user_id, character_id)
    return [ChatListItem(**chat) for chat in chats]


@router.post("", response_model=CreateChatResponse, status_code=201)
async def create_chat(request: Request, body: CreateChatRequest) -> CreateChatResponse:
    """Create a new chat session with LLM-generated title.
    创建新聊天会话，使用 LLM 生成标题。

    Args:
        body: Request body containing character_id and first_message.
              请求体，包含 character_id 和 first_message。

    Returns:
        Created chat metadata.
        创建的聊天元数据。
    """
    storage = request.app.state.storage
    config = request.app.state.config
    user_id = "default"  # Phase 5: hardcoded user_id

    # Generate title with LLM (with fallback)
    # 使用 LLM 生成标题（带降级）
    llm_config = config.get("llm", {})
    title = await _generate_title_with_llm(body.first_message, llm_config)

    if not title:
        title = _fallback_title(body.first_message)
        logger.info(f"Using fallback title: {title}")
    else:
        logger.info(f"Generated title: {title}")

    # Create chat in storage
    # 在存储中创建聊天
    chat_meta = await storage.create_chat(user_id, body.character_id, title)

    return CreateChatResponse(
        id=chat_meta["id"],
        title=chat_meta["title"],
        character_id=chat_meta["character_id"],
        created_at=chat_meta["created_at"],
    )


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(
    request: Request,
    chat_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
    offset: int = Query(0, ge=0, description="Message offset for pagination"),
) -> ChatDetailResponse:
    """Get chat details with message history.
    获取聊天详情及消息历史。

    Args:
        chat_id: Chat session ID.
                 聊天会话 ID。
        limit: Maximum number of messages to return.
               返回的最大消息数。
        offset: Message offset for pagination.
                分页的消息偏移量。

    Returns:
        Chat metadata and messages.
        聊天元数据和消息。

    Raises:
        HTTPException: 404 if chat not found.
                       聊天不存在时返回 404。
    """
    storage = request.app.state.storage

    # Get chat metadata
    # 获取聊天元数据
    chat_meta = await storage.get_chat(chat_id)
    if not chat_meta:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")

    # Get messages
    # 获取消息
    messages = await storage.get_messages(chat_id, limit=limit, offset=offset)

    return ChatDetailResponse(
        metadata=ChatListItem(**chat_meta),
        messages=[ChatMessage(**msg) for msg in messages],
    )


@router.post("/{chat_id}/update", response_model=ChatListItem)
async def update_chat_title(
    request: Request, chat_id: str, body: UpdateChatTitleRequest
) -> ChatListItem:
    """Update chat title.
    更新聊天标题。

    Args:
        chat_id: Chat session ID.
                 聊天会话 ID。
        body: Request body containing new title.
              请求体，包含新标题。

    Returns:
        Updated chat metadata.
        更新后的聊天元数据。

    Raises:
        HTTPException: 404 if chat not found.
                       聊天不存在时返回 404。
    """
    storage = request.app.state.storage

    try:
        updated_chat = await storage.update_chat(chat_id, title=body.title)
        logger.info(f"Updated chat {chat_id} title to: {body.title}")
        return ChatListItem(**updated_chat)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/{chat_id}/delete", status_code=204)
async def delete_chat(request: Request, chat_id: str) -> None:
    """Delete chat session.
    删除聊天会话。

    Args:
        chat_id: Chat session ID.
                 聊天会话 ID。

    Raises:
        HTTPException: 404 if chat not found.
                       聊天不存在时返回 404。
    """
    storage = request.app.state.storage

    try:
        await storage.delete_chat(chat_id)
        logger.info(f"Deleted chat {chat_id}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
