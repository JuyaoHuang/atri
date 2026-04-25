# ASR Module - Test Execution

## Offline Checks

```powershell
cd D:\Coding\GitHub_Resuorse\emotion-robot\atri
uv run ruff check src tests/routes/test_asr.py
uv run python -m mypy src/ --ignore-missing-imports
uv run pytest tests/routes/test_asr.py -v
```

Expected:
- Ruff: `All checks passed!`
- Mypy: `Success: no issues found`
- Pytest: `5 passed`

## API Smoke Test

Start the backend:

```powershell
cd D:\Coding\GitHub_Resuorse\emotion-robot\atri
uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8430
```

Query ASR state:

```powershell
Invoke-RestMethod http://127.0.0.1:8430/api/asr/providers
Invoke-RestMethod http://127.0.0.1:8430/api/asr/config
Invoke-RestMethod http://127.0.0.1:8430/api/asr/health
```

Expected:
- Providers include `web_speech_api`, `faster_whisper`, `whisper_cpp`, and `openai_whisper`.
- `web_speech_api` is available for browser streaming and rejects backend transcription.
- Missing optional local dependencies are reported as unavailable without failing startup.

## Backend Transcription Smoke Test

Prerequisites:
- Select a backend transcription provider in `config/asr_config.yaml`.
- Install the provider dependency and make the model/API key available.
- Prepare a short WAV/WebM/OGG recording.

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8430/api/asr/transcribe?provider=faster_whisper" `
  -Form @{ audio = Get-Item "D:\path\to\recording.wav" }
```

Expected:
- Response contains `provider` and non-empty `text`.
- If the provider cannot run, the API returns a clear 503 error.
