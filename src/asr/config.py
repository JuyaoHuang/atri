"""ASR configuration loading and persistence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ATRI_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASR_CONFIG_PATH = _ATRI_ROOT / "config" / "asr_config.yaml"
SENSITIVE_CONFIG_KEYS = {"api_key", "token", "secret", "password"}

DEFAULT_ASR_CONFIG: dict[str, Any] = {
    "asr_model": "web_speech_api",
    "auto_send": {
        "enabled": False,
        "delay_ms": 2000,
    },
    "web_speech_api": {
        "language": "zh-CN",
        "continuous": True,
        "interim_results": True,
        "max_alternatives": 1,
    },
    "faster_whisper": {
        "model_path": "distil-medium.en",
        "download_root": "models/whisper",
        "language": "en",
        "device": "auto",
        "compute_type": "int8",
        "prompt": "",
    },
    "whisper_cpp": {
        "model_name": "small",
        "model_dir": "models/whisper",
        "print_realtime": False,
        "print_progress": False,
        "language": "auto",
        "prompt": "",
    },
    "whisper": {
        "name": "medium",
        "download_root": "models/whisper",
        "device": "cpu",
        "prompt": "",
    },
    "openai_whisper": {
        "model": "whisper-1",
        "api_key": "${OPENAI_API_KEY}",
        "base_url": "",
        "language": "",
        "prompt": "",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mapping values without mutating inputs."""

    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class ASRConfigStore:
    """Small YAML-backed configuration store for ASR settings."""

    def __init__(
        self,
        initial_config: dict[str, Any] | None = None,
        *,
        path: Path | None = None,
    ) -> None:
        self.path = path or DEFAULT_ASR_CONFIG_PATH
        raw_config = self._read_raw_config()
        self._persist_config = deep_merge(
            DEFAULT_ASR_CONFIG,
            raw_config if raw_config is not None else initial_config or {},
        )
        self._config = deep_merge(
            DEFAULT_ASR_CONFIG,
            initial_config or raw_config or {},
        )

    def read(self) -> dict[str, Any]:
        """Return a defensive copy of the current ASR config."""

        return deepcopy(self._config)

    def update(self, patch: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
        """Merge a partial config update and persist it by default."""

        self._config = deep_merge(self._config, patch)
        self._persist_config = deep_merge(self._persist_config, patch)
        if persist:
            self.save()
        return self.read()

    def replace(self, config: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
        """Replace the current config after applying defaults."""

        self._config = deep_merge(DEFAULT_ASR_CONFIG, config)
        self._persist_config = deep_merge(DEFAULT_ASR_CONFIG, config)
        if persist:
            self.save()
        return self.read()

    def save(self) -> None:
        """Persist the current config to YAML."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(
                self._config_for_save(self._persist_config, DEFAULT_ASR_CONFIG),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _read_raw_config(self) -> dict[str, Any] | None:
        """Read the persisted ASR YAML without environment substitution."""

        if not self.path.is_file():
            return None
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}

    def _config_for_save(
        self,
        config: dict[str, Any],
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Return config safe to persist to disk."""

        safe = deepcopy(config)
        for key, value in safe.items():
            default_value = defaults.get(key)
            if isinstance(value, dict) and isinstance(default_value, dict):
                safe[key] = self._config_for_save(value, default_value)
            elif key.lower() in SENSITIVE_CONFIG_KEYS and self._is_env_placeholder(default_value):
                safe[key] = default_value
        return safe

    def _is_env_placeholder(self, value: Any) -> bool:
        return isinstance(value, str) and value.startswith("${") and value.endswith("}")
