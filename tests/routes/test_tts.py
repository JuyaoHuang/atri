"""Tests for Phase 10 TTS routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient

from src.app import create_app
from src.tts import TTSConfigStore, TTSService
from src.tts.providers import cosyvoice3_tts as cosyvoice3_module
from src.tts.providers import edge_tts as edge_tts_module
from src.tts.providers import gpt_sovits_tts as gpt_sovits_module
from src.utils.config_loader import load_config


@pytest_asyncio.fixture
async def client_and_config_path(tmp_path: Path):
    """Create test client with isolated TTS config persistence."""

    config = load_config("config.yaml")
    app = create_app(config)
    config_path = tmp_path / "tts_config.yaml"
    app.state.tts_service = TTSService(
        TTSConfigStore(
            {
                "tts_model": "edge_tts",
                "enabled": False,
                "auto_play": False,
                "show_player_on_home": False,
                "volume": 1.0,
            },
            path=config_path,
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, config_path


@pytest.mark.asyncio
async def test_list_tts_providers_returns_registered_statuses(client_and_config_path):
    client, _config_path = client_and_config_path

    response = await client.get("/api/tts/providers")

    assert response.status_code == 200
    providers = response.json()
    names = {provider["name"] for provider in providers}
    assert {"edge_tts", "gpt_sovits_tts", "siliconflow_tts", "cosyvoice3_tts"} <= names

    edge = next(provider for provider in providers if provider["name"] == "edge_tts")
    assert edge["active"] is True
    assert edge["supports_streaming"] is False
    assert edge["media_type"] == "audio/mpeg"


@pytest.mark.asyncio
async def test_update_tts_config_persists_olv_shaped_config(client_and_config_path):
    client, config_path = client_and_config_path

    response = await client.put(
        "/api/tts/config",
        json={
            "tts_model": "gpt_sovits_tts",
            "enabled": True,
            "auto_play": True,
            "show_player_on_home": True,
            "volume": 0.8,
            "gpt_sovits_tts": {
                "api_url": "http://127.0.0.1:9880/tts",
                "timeout_seconds": 90,
                "media_type": "wav",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()["config"]
    assert data["tts_model"] == "gpt_sovits_tts"
    assert data["enabled"] is True
    assert data["auto_play"] is True
    assert data["show_player_on_home"] is True
    assert data["volume"] == 0.8
    assert "media_type" not in data.get("gpt_sovits_tts", {})

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["tts_model"] == "gpt_sovits_tts"
    assert persisted["auto_play"] is True
    assert persisted["show_player_on_home"] is True
    assert "gpt_sovits_tts" not in persisted


def test_tts_config_store_does_not_backfill_defaults_on_save(tmp_path: Path):
    config_path = tmp_path / "tts_config.yaml"
    store = TTSConfigStore({"tts_model": "edge_tts"}, path=config_path)

    store.update({"auto_play": True})

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted == {"tts_model": "edge_tts", "auto_play": True}


def test_tts_config_store_preserves_raw_secret_placeholder_on_save(tmp_path: Path):
    config_path = tmp_path / "tts_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tts_model: siliconflow_tts",
                "siliconflow_tts:",
                "  api_key: ${SILICONFLOW_API_KEY}",
                "  api_url: https://api.siliconflow.cn/v1/audio/speech",
                "",
            ]
        ),
        encoding="utf-8",
    )

    store = TTSConfigStore(
        {
            "tts_model": "siliconflow_tts",
            "siliconflow_tts": {
                "api_key": "resolved-runtime-key",
                "api_url": "https://api.siliconflow.cn/v1/audio/speech",
            },
        },
        path=config_path,
    )

    assert store.read()["siliconflow_tts"]["api_key"] == "resolved-runtime-key"

    store.update({"auto_play": True})

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["siliconflow_tts"]["api_key"] == "${SILICONFLOW_API_KEY}"
    assert persisted["auto_play"] is True


def test_tts_service_masks_secret_values_in_public_config(tmp_path: Path):
    service = TTSService(
        TTSConfigStore(
            {
                "tts_model": "siliconflow_tts",
                "siliconflow_tts": {
                    "api_key": "resolved-runtime-key",
                    "api_url": "https://api.siliconflow.cn/v1/audio/speech",
                },
            },
            path=tmp_path / "tts_config.yaml",
        )
    )

    config = service.get_config()
    providers = service.list_providers()

    assert config["siliconflow_tts"]["api_key"] == "********"
    siliconflow_provider = next(
        provider for provider in providers if provider["name"] == "siliconflow_tts"
    )
    assert siliconflow_provider["config"]["api_key"] == "********"


def test_tts_service_ignores_masked_secret_patch(tmp_path: Path):
    config_path = tmp_path / "tts_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tts_model: siliconflow_tts",
                "siliconflow_tts:",
                "  api_key: ${SILICONFLOW_API_KEY}",
                "  api_url: https://api.siliconflow.cn/v1/audio/speech",
                "",
            ]
        ),
        encoding="utf-8",
    )
    service = TTSService(
        TTSConfigStore(
            {
                "tts_model": "siliconflow_tts",
                "siliconflow_tts": {
                    "api_key": "resolved-runtime-key",
                    "api_url": "https://api.siliconflow.cn/v1/audio/speech",
                    "default_voice": "old-voice",
                },
            },
            path=config_path,
        )
    )

    service.update_config(
        {
            "siliconflow_tts": {
                "api_key": "********",
                "default_voice": "new-voice",
            }
        }
    )

    raw_config = service.config_store.read()
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw_config["siliconflow_tts"]["api_key"] == "resolved-runtime-key"
    assert raw_config["siliconflow_tts"]["default_voice"] == "new-voice"
    assert persisted["siliconflow_tts"]["api_key"] == "${SILICONFLOW_API_KEY}"
    assert persisted["siliconflow_tts"]["default_voice"] == "new-voice"


def test_tts_service_blocks_provider_write_protected_fields(tmp_path: Path):
    config_path = tmp_path / "tts_config.yaml"
    service = TTSService(
        TTSConfigStore(
            {
                "tts_model": "edge_tts",
            },
            path=config_path,
        )
    )

    service.update_config(
        {
            "edge_tts": {
                "voice": "en-US-AriaNeural",
                "rate": "+10%",
                "pitch": "+10Hz",
            },
            "gpt_sovits_tts": {
                "api_url": "http://127.0.0.1:9880/tts",
                "timeout_seconds": 60,
                "text_lang": "en",
                "ref_audio_path": "bad.wav",
                "prompt_text": "bad prompt",
                "media_type": "mp3",
            },
            "siliconflow_tts": {
                "api_key": "bad-key",
                "api_url": "https://bad.example/v1/audio/speech",
                "default_model": "bad-model",
                "default_voice": "FunAudioLLM/CosyVoice2-0.5B:claire",
                "response_format": "wav",
                "speed": 1.1,
                "stream": True,
            },
            "cosyvoice3_tts": {
                "client_url": "http://127.0.0.1:50000/",
                "sft_dropdown": "zh-female",
                "prompt_wav_upload_url": "bad.wav",
                "prompt_wav_record_url": "bad.wav",
                "stream": True,
                "speed": 1.2,
            },
        }
    )

    config = service.config_store.read()
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["edge_tts"]["voice"] == "en-US-AriaNeural"
    assert "pitch" not in config["edge_tts"]
    assert persisted["edge_tts"] == {"voice": "en-US-AriaNeural", "rate": "+10%"}
    assert "gpt_sovits_tts" not in persisted
    assert "api_key" not in config["siliconflow_tts"]
    assert "api_url" not in config["siliconflow_tts"]
    assert "default_model" not in config["siliconflow_tts"]
    assert "response_format" not in config["siliconflow_tts"]
    assert persisted["siliconflow_tts"] == {
        "default_voice": "FunAudioLLM/CosyVoice2-0.5B:claire",
        "stream": True,
    }
    assert persisted["cosyvoice3_tts"] == {
        "sft_dropdown": "zh-female",
        "stream": True,
        "speed": 1.2,
    }


def test_tts_config_store_preserves_manual_disk_edits_before_save(tmp_path: Path):
    config_path = tmp_path / "tts_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tts_model: edge_tts",
                "gpt_sovits_tts:",
                "  api_url: http://127.0.0.1:9880/tts",
                "  ref_audio_path: old.wav",
                "  prompt_text: old prompt",
                "",
            ]
        ),
        encoding="utf-8",
    )
    store = TTSConfigStore(path=config_path)

    config_path.write_text(
        "\n".join(
            [
                "tts_model: gpt_sovits_tts",
                "gpt_sovits_tts:",
                "  api_url: http://127.0.0.1:9880/tts",
                "  ref_audio_path: manual.wav",
                "  prompt_text: manual prompt",
                "  text_lang: zh",
                "",
            ]
        ),
        encoding="utf-8",
    )

    store.update({"auto_play": True})

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["tts_model"] == "gpt_sovits_tts"
    assert persisted["auto_play"] is True
    assert persisted["gpt_sovits_tts"]["ref_audio_path"] == "manual.wav"
    assert persisted["gpt_sovits_tts"]["prompt_text"] == "manual prompt"


def test_tts_config_store_patches_values_without_reformatting_yaml(tmp_path: Path):
    config_path = tmp_path / "tts_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "# keep header comment",
                "tts_model: edge_tts # active provider",
                "edge_tts:",
                "  voice: zh-CN-XiaoxiaoNeural # untouched",
                "  rate: +0% # writable",
                "gpt_sovits_tts:",
                "  prompt_lang: 'zh'",
                "  prompt_text: \"hello\"",
                "siliconflow_tts:",
                "  default_voice: 'old-voice' # keep quote and comment",
                "  stream: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    store = TTSConfigStore(path=config_path)

    store.update(
        {
            "tts_model": "siliconflow_tts",
            "edge_tts": {"rate": "+10%"},
            "siliconflow_tts": {
                "default_voice": "new voice",
                "stream": True,
            },
        }
    )

    updated = config_path.read_text(encoding="utf-8")
    assert "# keep header comment" in updated
    assert "tts_model: siliconflow_tts # active provider" in updated
    assert "  voice: zh-CN-XiaoxiaoNeural # untouched" in updated
    assert "  rate: +10% # writable" in updated
    assert "  prompt_lang: 'zh'" in updated
    assert '  prompt_text: "hello"' in updated
    assert "  default_voice: 'new voice' # keep quote and comment" in updated
    assert "  stream: true" in updated


@pytest.mark.asyncio
async def test_switch_tts_provider_rejects_unknown_provider(client_and_config_path):
    client, _config_path = client_and_config_path

    response = await client.post("/api/tts/switch", json={"provider": "unknown"})

    assert response.status_code == 400
    assert "Unknown TTS provider" in response.json()["detail"]


@pytest.mark.asyncio
async def test_missing_optional_edge_tts_dependency_returns_503(
    client_and_config_path,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _config_path = client_and_config_path
    original_find_spec = edge_tts_module.importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "edge_tts":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(edge_tts_module.importlib.util, "find_spec", fake_find_spec)

    response = await client.post("/api/tts/synthesize", json={"text": "hello"})

    assert response.status_code == 503
    assert "edge-tts" in response.json()["detail"]


@pytest.mark.asyncio
async def test_gpt_sovits_synthesize_returns_audio_response(
    client_and_config_path,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _config_path = client_and_config_path

    class DummyResponse:
        status_code = 200
        content = b"wav-data"
        text = ""

    class DummyAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.request_params: dict[str, Any] | None = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, url: str, *, params: dict[str, Any]):
            assert url == "http://127.0.0.1:9880/tts"
            assert params["text"] == "hello"
            assert params["media_type"] == "wav"
            return DummyResponse()

    monkeypatch.setattr(gpt_sovits_module.httpx, "AsyncClient", DummyAsyncClient)

    switch_response = await client.post("/api/tts/switch", json={"provider": "gpt_sovits_tts"})
    assert switch_response.status_code == 200

    response = await client.post("/api/tts/synthesize", json={"text": "hello"})

    assert response.status_code == 200
    assert response.content == b"wav-data"
    assert response.headers["x-tts-provider"] == "gpt_sovits_tts"
    assert response.headers["content-type"].startswith("audio/wav")


@pytest.mark.asyncio
async def test_cosyvoice3_synthesize_reads_gradio_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    audio_path = tmp_path / "result.wav"
    audio_path.write_bytes(b"wav-data")
    calls: dict[str, Any] = {}

    class DummyClient:
        def __init__(self, client_url: str) -> None:
            calls["client_url"] = client_url

        def predict(self, **kwargs: Any) -> str:
            calls.update(kwargs)
            return str(audio_path)

    def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> object | None:
        if name == "gradio_client":
            return object()
        return None

    def fake_handle_file(value: str) -> str:
        return f"file:{value}"

    monkeypatch.setattr(cosyvoice3_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        cosyvoice3_module,
        "_load_gradio_client",
        lambda: (DummyClient, fake_handle_file),
    )

    provider = cosyvoice3_module.CosyVoice3TTSProvider(
        client_url="http://127.0.0.1:50000/",
        sft_dropdown="voice-a",
        prompt_wav_upload_url="upload.wav",
        prompt_wav_record_url="record.wav",
    )

    audio = await provider.synthesize("hello", voice_id="voice-b", stream=True, speed=1.25)

    assert audio == b"wav-data"
    assert calls["client_url"] == "http://127.0.0.1:50000/"
    assert calls["tts_text"] == "hello"
    assert calls["sft_dropdown"] == "voice-b"
    assert calls["prompt_wav_upload"] == "file:upload.wav"
    assert calls["prompt_wav_record"] == "file:record.wav"
    assert calls["stream"] is True
    assert calls["speed"] == 1.25


@pytest.mark.asyncio
async def test_get_tts_voices_returns_provider_voice_metadata(client_and_config_path):
    client, _config_path = client_and_config_path

    response = await client.get("/api/tts/voices", params={"provider": "gpt_sovits_tts"})

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "gpt_sovits_tts"
    assert data["voices"][0]["id"] == "default"
