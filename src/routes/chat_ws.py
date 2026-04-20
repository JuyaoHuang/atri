"""WebSocket endpoint for real-time chat streaming.
WebSocket 实时聊天流式端点。

Bridges ChatAgent streaming output to frontend via WebSocket protocol.
通过 WebSocket 协议桥接 ChatAgent 流式输出到前端。

Message Protocol (参考 airi 事件命名):
消息协议（参考 airi 事件命名）：

Client → Server:
  - {"type": "input:text", "data": {"text": "...", "chat_id": "...",
     "character_id": "..."}}
  - {"type": "ping"}

Server → Client:
  - {"type": "output:chat:chunk", "data": {"chunk": "...", "chat_id": "...",
     "character_id": "..."}}
  - {"type": "output:chat:complete", "data": {"full_reply": "...",
     "chat_id": "...", "character_id": "..."}}
  - {"type": "error", "data": {"message": "...", "chat_id": "..."}}
  - {"type": "pong"}

Reference: docs/Phase5_执行规格.md §US-SRV-006, docs/OLV架构文档.md
"""

import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from starlette.websockets import WebSocketState

from src.llm.exceptions import LLMError


async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint handler for chat streaming.
    WebSocket 聊天流式端点处理器。

    Args:
        websocket: FastAPI WebSocket connection.
                   FastAPI WebSocket 连接。
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Access app state (ServiceContext + Storage)
    # 访问 app state（ServiceContext + Storage）
    service_context = websocket.app.state.service_context
    storage = websocket.app.state.storage

    try:
        while True:
            # Receive message from client
            # 接收客户端消息
            raw_message = await websocket.receive_text()

            # Parse JSON
            # 解析 JSON
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received: {e}")
                await _send_error(websocket, "Invalid JSON format", chat_id=None)
                continue

            # Extract message type
            # 提取消息类型
            msg_type = message.get("type")
            if not msg_type:
                logger.warning("Message missing 'type' field")
                await _send_error(websocket, "Message missing 'type' field", chat_id=None)
                continue

            # Route message by type
            # 按类型路由消息
            if msg_type == "ping":
                await _handle_ping(websocket)
            elif msg_type == "input:text":
                await _handle_text_input(websocket, message, service_context, storage)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await _send_error(
                    websocket,
                    f"Unknown message type: {msg_type}",
                    chat_id=message.get("data", {}).get("chat_id"),
                )

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await _send_error(websocket, f"Internal server error: {e}", chat_id=None)
        except Exception:
            pass  # Connection already closed
    finally:
        logger.info("WebSocket connection cleanup complete")


async def _handle_ping(websocket: WebSocket) -> None:
    """Handle ping message (heartbeat).
    处理 ping 消息（心跳）。

    Args:
        websocket: WebSocket connection.
                   WebSocket 连接。
    """
    await websocket.send_json({"type": "pong"})


async def _handle_text_input(
    websocket: WebSocket,
    message: dict[str, Any],
    service_context: Any,
    storage: Any,
) -> None:
    """Handle text input message and stream ChatAgent response.
    处理文本输入消息并流式传输 ChatAgent 响应。

    Args:
        websocket: WebSocket connection.
                   WebSocket 连接。
        message: Parsed message dict with 'data' field.
                 解析后的消息字典，含 'data' 字段。
        service_context: ServiceContext instance from app.state.
                         来自 app.state 的 ServiceContext 实例。
        storage: ChatStorage instance from app.state.
                 来自 app.state 的 ChatStorage 实例。
    """
    data = message.get("data", {})
    text = data.get("text")
    chat_id = data.get("chat_id")
    character_id = data.get("character_id")

    # Validate required fields
    # 验证必填字段
    if not text:
        await _send_error(websocket, "Missing 'text' field", chat_id=chat_id)
        return
    if not chat_id:
        await _send_error(websocket, "Missing 'chat_id' field", chat_id=None)
        return
    if not character_id:
        await _send_error(websocket, "Missing 'character_id' field", chat_id=chat_id)
        return

    logger.info(f"Received text input | chat_id={chat_id} | character_id={character_id}")

    # Get or create ChatAgent for this character/user
    # 获取或创建此 character/user 的 ChatAgent
    user_id = "default"  # Phase 5: hardcoded user_id (D7)
    try:
        agent = service_context.get_or_create_agent(character_id, user_id)
    except Exception as e:
        logger.error(f"Failed to get ChatAgent: {e}")
        await _send_error(
            websocket,
            f"Failed to initialize character '{character_id}': {e}",
            chat_id=chat_id,
        )
        return

    # Stream ChatAgent response
    # 流式传输 ChatAgent 响应
    chunks = []
    try:
        async for chunk in agent.chat(text):
            chunks.append(chunk)
            # Send chunk to client
            # 发送 chunk 给客户端
            await websocket.send_json(
                {
                    "type": "output:chat:chunk",
                    "data": {
                        "chunk": chunk,
                        "chat_id": chat_id,
                        "character_id": character_id,
                    },
                }
            )

        # Stream complete
        # 流式传输完成
        full_reply = "".join(chunks)

        # Persist messages to storage BEFORE sending complete event
        # 在发送完成事件之前持久化消息到存储
        # This ensures messages are saved before client closes connection
        # 这确保在客户端关闭连接前消息已保存
        user_id = "default"  # Phase 5: hardcoded user_id (D7)
        try:
            logger.debug(f"Starting message persistence | chat_id={chat_id}")
            await storage.append_message(chat_id, "human", text, name=user_id)
            await storage.append_message(chat_id, "ai", full_reply, name=character_id)
            logger.debug(f"Messages persisted | chat_id={chat_id}")
        except ValueError as e:
            # Chat not found (client may have deleted it)
            # 聊天不存在（客户端可能已删除）
            logger.warning(f"Failed to persist messages: {e}")
        except Exception as e:
            logger.error(f"Unexpected error persisting messages: {e}")

        # Now send complete event
        # 现在发送完成事件
        await websocket.send_json(
            {
                "type": "output:chat:complete",
                "data": {
                    "full_reply": full_reply,
                    "chat_id": chat_id,
                    "character_id": character_id,
                },
            }
        )

        logger.info(f"Chat complete | chat_id={chat_id} | reply_length={len(full_reply)}")

    except LLMError as e:
        # LLM error path (already handled by ChatAgent, but catch here for safety)
        # LLM 错误路径（ChatAgent 已处理，但此处捕获以确保安全）
        logger.error(f"LLM error during chat: {e}")
        await _send_error(
            websocket,
            f"LLM call failed: {e}",
            chat_id=chat_id,
        )
    except Exception as e:
        logger.error(f"Unexpected error during chat: {e}")
        await _send_error(
            websocket,
            f"Chat processing failed: {e}",
            chat_id=chat_id,
        )


async def _send_error(websocket: WebSocket, message: str, chat_id: str | None) -> None:
    """Send error message to client.
    向客户端发送错误消息。

    Args:
        websocket: WebSocket connection.
                   WebSocket 连接。
        message: Error message.
                 错误消息。
        chat_id: Optional chat ID for context.
                 可选的聊天 ID（用于上下文）。
    """
    error_data: dict[str, Any] = {"message": message}
    if chat_id:
        error_data["chat_id"] = chat_id

    await websocket.send_json({"type": "error", "data": error_data})
