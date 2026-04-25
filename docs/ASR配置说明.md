# ASR 配置说明

> **适用范围**: Phase 9 ASR 语音输入  
> **配置文件**: `atri/config/asr_config.yaml`  
> **设置页面**: `atri-webui` 的 `/settings/modules/hearing`  
> **最后更新**: 2026-04-25

本文说明 ASR（Automatic Speech Recognition，自动语音识别）的配置结构、Provider 选择、前端设置页对应关系和常见排障方法。

---

## 1. 快速开始

默认配置使用浏览器内置的 Web Speech API。它不需要后端模型，也不需要 API Key。

```yaml
asr_model: web_speech_api
auto_send:
  enabled: false
  delay_ms: 2000
```

启动后打开前端：

```text
/settings/modules/hearing
```

你可以在该页面完成以下操作：

- 选择麦克风输入设备
- 切换 ASR Provider
- 配置识别语言、模型和路径
- 测试语音转文字
- 配置转写后是否自动发送到聊天

---

## 2. 配置加载方式

根配置文件 `atri/config.yaml` 通过下面的入口引用 ASR 配置：

```yaml
asr_config: config/asr_config.yaml
```

后端 `config_loader` 会把它加载到运行时配置的 `asr` 节点下：

```python
config["asr"]
```

因此 ASR 模块实际读取的是：

```text
atri/config/asr_config.yaml -> runtime config["asr"]
```

---

## 3. 顶层配置结构

当前配置采用 Open-LLM-VTuber 风格：`asr_model` 指定当前 Provider，Provider 参数使用同名顶层块保存。

```yaml
asr_model: web_speech_api
auto_send:
  enabled: false
  delay_ms: 2000

web_speech_api:
  language: zh-CN
  continuous: true
  interim_results: true
  max_alternatives: 1

faster_whisper:
  model_path: distil-medium.en
  download_root: models/whisper
  language: en
  device: auto
  compute_type: int8
  prompt: ''

whisper_cpp:
  model_name: small
  model_dir: models/whisper
  print_realtime: false
  print_progress: false
  language: auto
  prompt: ''

openai_whisper:
  model: whisper-1
  api_key: ${OPENAI_API_KEY}
  base_url: ''
  language: ''
  prompt: ''
```

### 顶层字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `asr_model` | string | 当前启用的 ASR Provider 名称。 |
| `auto_send.enabled` | boolean | 转写完成后是否自动发送到聊天输入链路。默认关闭。 |
| `auto_send.delay_ms` | number | 自动发送延迟，单位毫秒。默认 `2000`。 |

当前可选的 `asr_model`：

| Provider | 类型 | 后端转写 | 浏览器流式 | 说明 |
| --- | --- | --- | --- | --- |
| `web_speech_api` | 浏览器 | 否 | 是 | 使用浏览器 SpeechRecognition。默认推荐。 |
| `faster_whisper` | 本地 | 是 | 否 | 本地 faster-whisper，参考 OLV 链路。 |
| `whisper_cpp` | 本地 | 是 | 否 | 本地 pywhispercpp，参考 OLV 链路。 |
| `openai_whisper` | 云服务 | 是 | 否 | OpenAI-compatible 音频转写。 |

`whisper` 配置块目前保留给 OLV 兼容和后续扩展。Phase 9 当前没有注册 `whisper` Provider，不要把 `asr_model` 设置为 `whisper`。

---

## 4. Provider 配置

### 4.1 `web_speech_api`

浏览器端语音识别 Provider。后端只保存配置和状态，不接收音频转写。

```yaml
asr_model: web_speech_api
web_speech_api:
  language: zh-CN
  continuous: true
  interim_results: true
  max_alternatives: 1
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `language` | string | `zh-CN` | 浏览器识别语言。常用值：`zh-CN`、`en-US`、`ja-JP`。 |
| `continuous` | boolean | `true` | 是否持续监听。 |
| `interim_results` | boolean | `true` | 是否返回临时识别结果。 |
| `max_alternatives` | number | `1` | 浏览器返回的候选结果数量。 |

使用建议：

- Chrome、Edge、Safari 对 Web Speech API 支持较好。
- Firefox 通常不可用或支持有限。
- 该 Provider 不需要 Python 依赖和模型文件。

### 4.2 `faster_whisper`

本地 faster-whisper Provider。适合本地离线转写。

```yaml
asr_model: faster_whisper
faster_whisper:
  model_path: distil-medium.en
  download_root: models/whisper
  language: en
  device: auto
  compute_type: int8
  prompt: ''
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_path` | string | `distil-medium.en` | 模型名称、本地路径或 Hugging Face 模型 ID。 |
| `download_root` | string | `models/whisper` | 模型下载和缓存目录。 |
| `language` | string | `en` | 识别语言。空字符串或 `auto` 表示自动检测。 |
| `device` | string | `auto` | 推理设备。常用值：`auto`、`cpu`、`cuda`。 |
| `compute_type` | string | `int8` | 推理精度。CPU 常用 `int8`。 |
| `prompt` | string | `''` | 初始提示词，可改善专有名词识别。 |

依赖说明：

```powershell
cd D:\Coding\GitHub_Resuorse\emotion-robot\atri
uv add faster-whisper
```

如果依赖或模型不可用，Provider 会显示为 unavailable，不会阻止后端启动。

### 4.3 `whisper_cpp`

本地 whisper.cpp Provider，使用 `pywhispercpp`。

```yaml
asr_model: whisper_cpp
whisper_cpp:
  model_name: small
  model_dir: models/whisper
  print_realtime: false
  print_progress: false
  language: auto
  prompt: ''
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_name` | string | `small` | pywhispercpp 模型名。常用值：`tiny`、`base`、`small`、`medium`。 |
| `model_dir` | string | `models/whisper` | 模型目录。 |
| `print_realtime` | boolean | `false` | 是否打印实时片段。 |
| `print_progress` | boolean | `false` | 是否打印进度。 |
| `language` | string | `auto` | 识别语言。 |
| `prompt` | string | `''` | 初始提示词。 |

依赖说明：

```powershell
cd D:\Coding\GitHub_Resuorse\emotion-robot\atri
uv add pywhispercpp
```

### 4.4 `openai_whisper`

OpenAI-compatible 云端转写 Provider。

```yaml
asr_model: openai_whisper
openai_whisper:
  model: whisper-1
  api_key: ${OPENAI_API_KEY}
  base_url: ''
  language: ''
  prompt: ''
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model` | string | `whisper-1` | 转写模型名。 |
| `api_key` | string | `${OPENAI_API_KEY}` | API Key。必须使用环境变量占位符。 |
| `base_url` | string | `''` | OpenAI-compatible endpoint。空字符串使用 SDK 默认值。 |
| `language` | string | `''` | 识别语言。空字符串表示自动检测。 |
| `prompt` | string | `''` | 初始提示词。 |

依赖说明：

```powershell
cd D:\Coding\GitHub_Resuorse\emotion-robot\atri
uv add openai
```

环境变量示例：

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
```

不要把真实 API Key 写进 `asr_config.yaml`。

---

## 5. 敏感配置规则

ASR 配置支持 `${ENV_NAME}` 环境变量占位符。`api_key` 等敏感字段必须使用占位符。

正确：

```yaml
openai_whisper:
  api_key: ${OPENAI_API_KEY}
```

错误：

```yaml
openai_whisper:
  api_key: sk-真实密钥
```

后端有两层保护：

- 运行时可以读取环境变量展开后的值。
- 保存 YAML 时会尽量保留占位符，不把运行时密钥写回配置文件。
- API 返回配置时会把 `api_key`、`token`、`secret`、`password` mask 成 `********`。

如果你发现真实 key 被写入配置文件，请立即：

1. 删除该 key，改回 `${OPENAI_API_KEY}`。
2. 轮换服务商后台的 API Key。
3. 扫描仓库，确认没有残留明文密钥。

---

## 6. 前端设置页对应关系

设置页路径：

```text
/settings/modules/hearing
```

页面区域和配置关系：

| 页面区域 | 对应配置 | 说明 |
| --- | --- | --- |
| Audio Input Device | 浏览器 localStorage | 选择麦克风设备。 |
| Providers | `asr_model` | 切换当前 Provider。 |
| Web Speech API 设置 | `web_speech_api` | 配置浏览器识别语言和临时结果。 |
| Faster Whisper 设置 | `faster_whisper` | 配置模型、语言和下载目录。 |
| Whisper.cpp 设置 | `whisper_cpp` | 配置模型名和模型目录。 |
| OpenAI Whisper 设置 | `openai_whisper` | 配置模型和 base URL。 |
| Auto-send Settings | `auto_send` | 控制转写后是否自动发送。 |
| Audio Monitor | 不落盘 | 测试麦克风音量和语音阈值。 |
| Transcription | 当前 Provider | 测试语音转文字链路。 |

---

## 7. 运行链路

### `web_speech_api`

```text
浏览器麦克风
  -> SpeechRecognition
  -> 前端得到 transcript
  -> 填入聊天输入框
  -> 用户手动发送，或按 auto_send 规则发送
```

后端不接收音频文件。

### 后端 Provider

适用于 `faster_whisper`、`whisper_cpp`、`openai_whisper`。

```text
浏览器麦克风
  -> MediaRecorder
  -> POST /api/asr/transcribe
  -> ASRService
  -> Provider 转写
  -> 返回 transcript
  -> 前端填入聊天输入框
```

---

## 8. API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/asr/providers` | 获取 Provider 列表、可用性和当前配置。 |
| `GET` | `/api/asr/config` | 获取当前 ASR 配置。敏感值会被 mask。 |
| `PUT` | `/api/asr/config` | 合并保存 ASR 配置。 |
| `POST` | `/api/asr/switch` | 切换当前 Provider。 |
| `GET` | `/api/asr/health` | 获取 ASR 健康状态。 |
| `POST` | `/api/asr/transcribe` | 上传音频并转写。仅后端 Provider 支持。 |

切换 Provider 示例：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8430/api/asr/switch" `
  -ContentType "application/json" `
  -Body '{"provider":"web_speech_api"}'
```

---

## 9. 常见配置示例

### 示例 A：中文浏览器识别

```yaml
asr_model: web_speech_api
web_speech_api:
  language: zh-CN
  continuous: true
  interim_results: true
  max_alternatives: 1
auto_send:
  enabled: false
  delay_ms: 2000
```

适合快速验证。推荐作为默认配置。

### 示例 B：本地 faster-whisper 英文识别

```yaml
asr_model: faster_whisper
faster_whisper:
  model_path: distil-medium.en
  download_root: models/whisper
  language: en
  device: auto
  compute_type: int8
  prompt: ''
```

适合离线英文转写。

### 示例 C：本地 whisper.cpp 自动语言识别

```yaml
asr_model: whisper_cpp
whisper_cpp:
  model_name: small
  model_dir: models/whisper
  print_realtime: false
  print_progress: false
  language: auto
  prompt: ''
```

适合轻量本地转写。

### 示例 D：云端 OpenAI-compatible 转写

```yaml
asr_model: openai_whisper
openai_whisper:
  model: whisper-1
  api_key: ${OPENAI_API_KEY}
  base_url: ''
  language: ''
  prompt: ''
```

适合不想在本地加载模型的场景。

---

## 10. 自检命令

后端 ASR 测试：

```powershell
cd D:\Coding\GitHub_Resuorse\emotion-robot\atri
uv run ruff check src tests/routes/test_asr.py
uv run python -m mypy src/ --ignore-missing-imports
uv run pytest tests/routes/test_asr.py -v
```

前端检查：

```powershell
cd D:\Coding\GitHub_Resuorse\emotion-robot\atri-webui
npm run type-check
npm run build
```

---

## 11. 常见问题

### Provider 显示 unavailable

原因通常是依赖未安装、模型路径错误或 API Key 未配置。

处理方式：

1. 查看 `/settings/modules/hearing` 中 Provider 卡片的状态提示。
2. 确认对应 Python 包已安装。
3. 确认模型目录或环境变量正确。

### Web Speech API 不可用

原因通常是浏览器不支持 `SpeechRecognition`。

处理方式：

- 使用 Chrome、Edge 或 Safari。
- 或切换到 `faster_whisper`、`whisper_cpp`、`openai_whisper`。

### 后端上传转写返回 503

如果当前 Provider 是 `web_speech_api`，这是预期行为。它只支持浏览器端识别。

处理方式：

- 切换到 `faster_whisper`、`whisper_cpp` 或 `openai_whisper`。
- 确认依赖和模型/API Key 可用。

### 转写成功但没有自动发送

默认行为是手动发送。语音转写会先填入聊天输入框，用户可以编辑后再发送。

如果需要自动发送：

1. 打开 `/settings/modules/hearing`。
2. 启用 **Auto-send transcribed text**。
3. 设置 `Auto-send delay`。

---

## 12. 修改建议

优先通过 `/settings/modules/hearing` 修改 ASR 配置。只有在以下场景才建议直接编辑 YAML：

- 初次配置环境变量占位符。
- 批量调整本地模型路径。
- 前端设置页无法启动，需要手工恢复默认 Provider。

直接编辑 YAML 后，重启后端服务让配置生效。
