"""Tests for Phase 9 ASR routes."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.asr import ASRConfigStore, ASRService
from src.asr.providers import faster_whisper as faster_whisper_module
from src.utils.config_loader import load_config


@pytest_asyncio.fixture
async def client_and_config_path(tmp_path: Path):
    """Create test client with isolated ASR config persistence."""

    config = load_config("config.yaml")
    app = create_app(config)
    config_path = tmp_path / "asr_config.yaml"
    app.state.asr_service = ASRService(
        ASRConfigStore(
            {
                "asr_model": "web_speech_api",
                "auto_send": {"enabled": False, "delay_ms": 2000},
            },
            path=config_path,
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, config_path


@pytest.mark.asyncio
async def test_list_asr_providers_returns_registered_statuses(client_and_config_path):
    client, _config_path = client_and_config_path

    response = await client.get("/api/asr/providers")

    assert response.status_code == 200
    providers = response.json()
    names = {provider["name"] for provider in providers}
    assert {"web_speech_api", "faster_whisper", "whisper_cpp", "openai_whisper"} <= names

    web_speech = next(provider for provider in providers if provider["name"] == "web_speech_api")
    assert web_speech["active"] is True
    assert web_speech["supports_backend_transcription"] is False
    assert web_speech["supports_browser_streaming"] is True


@pytest.mark.asyncio
async def test_update_asr_config_persists_olv_shaped_config(client_and_config_path):
    client, config_path = client_and_config_path

    response = await client.put(
        "/api/asr/config",
        json={
            "asr_model": "web_speech_api",
            "auto_send": {"enabled": True, "delay_ms": 1200},
            "web_speech_api": {"language": "en-US", "continuous": False},
        },
    )

    assert response.status_code == 200
    data = response.json()["config"]
    assert data["asr_model"] == "web_speech_api"
    assert data["auto_send"] == {"enabled": True, "delay_ms": 1200}
    assert data["web_speech_api"]["language"] == "en-US"
    assert data["web_speech_api"]["continuous"] is False

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["asr_model"] == "web_speech_api"
    assert persisted["auto_send"]["delay_ms"] == 1200


def test_asr_config_store_preserves_raw_secret_placeholder_on_save(tmp_path: Path):
    config_path = tmp_path / "asr_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "asr_model: web_speech_api",
                "openai_whisper:",
                "  model: whisper-1",
                "  api_key: ${OPENAI_API_KEY}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    store = ASRConfigStore(
        {
            "asr_model": "web_speech_api",
            "openai_whisper": {
                "model": "whisper-1",
                "api_key": "resolved-runtime-key",
            },
        },
        path=config_path,
    )

    assert store.read()["openai_whisper"]["api_key"] == "resolved-runtime-key"

    store.update({"auto_send": {"enabled": True, "delay_ms": 1500}})

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["openai_whisper"]["api_key"] == "${OPENAI_API_KEY}"
    assert persisted["auto_send"] == {"enabled": True, "delay_ms": 1500}


def test_asr_config_store_does_not_rewrite_unpatched_secret_on_save(tmp_path: Path):
    config_path = tmp_path / "asr_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "asr_model: web_speech_api",
                "openai_whisper:",
                "  model: whisper-1",
                "  api_key: resolved-runtime-key",
                "",
            ]
        ),
        encoding="utf-8",
    )

    store = ASRConfigStore(path=config_path)

    store.update({"auto_send": {"enabled": False, "delay_ms": 1000}})

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["openai_whisper"]["api_key"] == "resolved-runtime-key"


def test_asr_config_store_patches_values_without_reformatting_yaml(tmp_path: Path):
    config_path = tmp_path / "asr_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "# keep header comment",
                "asr_model: web_speech_api # active provider",
                "auto_send:",
                "  enabled: false # writable",
                "  delay_ms: 2000",
                "web_speech_api:",
                "  language: 'zh-CN' # keep quote",
                "  continuous: true",
                "openai_whisper:",
                "  api_key: ${OPENAI_API_KEY}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    store = ASRConfigStore(path=config_path)

    store.update(
        {
            "asr_model": "web_speech_api",
            "auto_send": {"enabled": True, "delay_ms": 1200},
            "web_speech_api": {"language": "en-US"},
        }
    )

    updated = config_path.read_text(encoding="utf-8")
    assert "# keep header comment" in updated
    assert "asr_model: web_speech_api # active provider" in updated
    assert "  enabled: true # writable" in updated
    assert "  delay_ms: 1200" in updated
    assert "  language: 'en-US' # keep quote" in updated
    assert "  continuous: true" in updated
    assert "  api_key: ${OPENAI_API_KEY}" in updated


def test_asr_service_masks_secret_values_in_public_config(tmp_path: Path):
    service = ASRService(
        ASRConfigStore(
            {
                "asr_model": "openai_whisper",
                "openai_whisper": {
                    "model": "whisper-1",
                    "api_key": "resolved-runtime-key",
                },
            },
            path=tmp_path / "asr_config.yaml",
        )
    )

    config = service.get_config()
    providers = service.list_providers()

    assert config["openai_whisper"]["api_key"] == "********"
    openai_provider = next(
        provider for provider in providers if provider["name"] == "openai_whisper"
    )
    assert openai_provider["config"]["api_key"] == "********"


def test_asr_service_ignores_masked_secret_patch(tmp_path: Path):
    config_path = tmp_path / "asr_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "asr_model: openai_whisper",
                "openai_whisper:",
                "  model: whisper-1",
                "  api_key: ${OPENAI_API_KEY}",
                "  base_url: ''",
                "",
            ]
        ),
        encoding="utf-8",
    )
    service = ASRService(
        ASRConfigStore(
            {
                "asr_model": "openai_whisper",
                "openai_whisper": {
                    "model": "whisper-1",
                    "api_key": "resolved-runtime-key",
                    "base_url": "",
                },
            },
            path=config_path,
        )
    )

    service.update_config(
        {
            "openai_whisper": {
                "model": "gpt-4o-mini-transcribe",
                "api_key": "********",
            }
        }
    )

    raw_config = service.config_store.read()
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw_config["openai_whisper"]["api_key"] == "resolved-runtime-key"
    assert raw_config["openai_whisper"]["model"] == "whisper-1"
    assert persisted["openai_whisper"]["api_key"] == "${OPENAI_API_KEY}"
    assert persisted["openai_whisper"]["model"] == "whisper-1"


def test_asr_service_blocks_provider_write_protected_fields(tmp_path: Path):
    config_path = tmp_path / "asr_config.yaml"
    service = ASRService(
        ASRConfigStore(
            {
                "asr_model": "faster_whisper",
                "faster_whisper": {
                    "model_path": "distil-large-v3",
                    "download_root": "models/whisper",
                    "language": "zh",
                },
                "whisper_cpp": {
                    "model_name": "small",
                    "model_dir": "models/whisper",
                },
                "openai_whisper": {
                    "model": "whisper-1",
                    "base_url": "",
                },
            },
            path=config_path,
        )
    )

    service.update_config(
        {
            "faster_whisper": {
                "model_path": "bad-model",
                "download_root": "bad-root",
                "language": "ja",
            },
            "whisper_cpp": {
                "model_name": "bad-model",
                "model_dir": "bad-dir",
            },
            "openai_whisper": {
                "model": "bad-model",
                "base_url": "https://bad.example/v1",
            },
        }
    )

    raw_config = service.config_store.read()
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw_config["faster_whisper"]["model_path"] == "distil-large-v3"
    assert raw_config["faster_whisper"]["download_root"] == "models/whisper"
    assert raw_config["faster_whisper"]["language"] == "ja"
    assert raw_config["whisper_cpp"]["model_name"] == "small"
    assert raw_config["whisper_cpp"]["model_dir"] == "models/whisper"
    assert raw_config["openai_whisper"]["model"] == "whisper-1"
    assert raw_config["openai_whisper"]["base_url"] == ""
    assert persisted["faster_whisper"]["model_path"] == "distil-large-v3"
    assert persisted["faster_whisper"]["download_root"] == "models/whisper"
    assert persisted["faster_whisper"]["language"] == "ja"
    assert persisted["whisper_cpp"]["model_name"] == "small"
    assert persisted["whisper_cpp"]["model_dir"] == "models/whisper"
    assert persisted["openai_whisper"]["model"] == "whisper-1"
    assert persisted["openai_whisper"]["base_url"] == ""


@pytest.mark.asyncio
async def test_switch_asr_provider_rejects_unknown_provider(client_and_config_path):
    client, _config_path = client_and_config_path

    response = await client.post("/api/asr/switch", json={"provider": "unknown"})

    assert response.status_code == 400
    assert "Unknown ASR provider" in response.json()["detail"]


@pytest.mark.asyncio
async def test_web_speech_api_rejects_backend_transcription(client_and_config_path):
    client, _config_path = client_and_config_path

    response = await client.post(
        "/api/asr/transcribe",
        files={"audio": ("recording.wav", b"not-used", "audio/wav")},
    )

    assert response.status_code == 503
    assert "does not support backend transcription" in response.json()["detail"]


@pytest.mark.asyncio
async def test_missing_optional_faster_whisper_dependency_returns_503(
    client_and_config_path,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _config_path = client_and_config_path
    original_find_spec = faster_whisper_module.importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "faster_whisper":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(faster_whisper_module.importlib.util, "find_spec", fake_find_spec)

    switch_response = await client.post("/api/asr/switch", json={"provider": "faster_whisper"})
    assert switch_response.status_code == 200

    response = await client.post(
        "/api/asr/transcribe",
        files={"audio": ("recording.wav", b"not-used", "audio/wav")},
    )

    assert response.status_code == 503
    assert "faster_whisper" in response.json()["detail"]
