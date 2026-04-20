"""
Factory function for creating chat storage instances.

Reads storage mode from config and returns appropriate implementation.
"""

from src.storage.interface import ChatStorageInterface


def create_chat_storage(config: dict) -> ChatStorageInterface:
    """
    Create chat storage instance based on config.

    Args:
        config: Full application config dict

    Returns:
        ChatStorageInterface implementation

    Raises:
        ValueError: If storage mode is unknown
        NotImplementedError: If database mode is requested (Phase 7)
    """
    storage_config = config.get("storage", {})
    mode = storage_config.get("mode", "json")

    if mode == "json":
        from src.storage.json_storage import JSONChatStorage

        base_path = storage_config["json"]["base_path"]
        return JSONChatStorage(base_path=base_path)
    elif mode == "database":
        raise NotImplementedError("Database storage is reserved for Phase 7")
    else:
        raise ValueError(f"Unknown storage mode: {mode}")
