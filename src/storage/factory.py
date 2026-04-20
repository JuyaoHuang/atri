"""Factory function for creating chat storage instances.

Reads storage mode from config and returns appropriate implementation.

从配置读取存储模式并返回相应的实现。

Reference: docs/Phase5_执行规格.md §US-SRV-001
"""

from src.storage.interface import ChatStorageInterface


def create_chat_storage(config: dict) -> ChatStorageInterface:
    """Create chat storage instance based on config.

    创建基于配置的聊天存储实例。

    Args:
        config: Storage config dict (from storage_config.yaml or config["storage"])
                存储配置字典（来自 storage_config.yaml 或 config["storage"]）

    Returns:
        ChatStorageInterface implementation
        ChatStorageInterface 实现

    Raises:
        ValueError: If storage mode is unknown
                   如果存储模式未知
        NotImplementedError: If database mode is requested (Phase 7)
                            如果请求数据库模式（Phase 7）
    """
    mode = config.get("mode", "json")

    if mode == "json":
        from src.storage.json_storage import JSONChatStorage

        json_config = config.get("json", {})
        base_path = json_config.get("base_path", "data/chats")
        return JSONChatStorage(base_path=base_path)
    elif mode == "database":
        raise NotImplementedError("Database storage is reserved for Phase 7")
    else:
        raise ValueError(f"Unknown storage mode: {mode}")
