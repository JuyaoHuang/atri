# atri

Emotion Robot 的后端服务。当前后端基于 FastAPI，提供聊天会话、角色管理、ASR、TTS、Live2D 资源管理、WebSocket 流式对话和 Phase 11 GitHub OAuth + JWT 认证。

## 功能

- FastAPI HTTP API，默认监听 `http://localhost:8430`
- WebSocket 实时对话，默认端点 `ws://localhost:8430/ws`
- 多角色管理，支持自定义角色和头像上传
- 多会话聊天历史，认证开启时按 GitHub 用户隔离数据
- ASR / TTS 提供商配置、切换和健康检查接口
- Live2D ZIP 模型上传、表情列表和静态资源托管
- 可选认证系统：本地部署可关闭，公网部署可启用 GitHub OAuth 白名单

## 环境要求

- Python `>= 3.11`
- `uv`
- 可用的 OpenAI 兼容 LLM 服务配置

## 快速开始

安装依赖：

```bash
uv sync
```

创建本地环境变量文件：

```bash
Copy-Item .env.example .env
```

编辑 `.env`，至少配置主聊天 LLM：

```env
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.2
```

启动后端：

```bash
uv run python -m src.main
```

健康检查：

```bash
curl http://localhost:8430/health
```

## 认证配置

认证配置位于 `config/auth.yaml`。本地开发或单机部署可以关闭认证：

```yaml
enabled: false
```

开启认证后，前端会把未登录用户路由到 `/login`，业务 HTTP 请求需要 `Authorization: Bearer <JWT_TOKEN>`，WebSocket 会使用 `?token=<JWT_TOKEN>`。

```yaml
enabled: true

jwt:
  secret_key: ${JWT_SECRET_KEY}
  algorithm: HS256
  expire_days: 7

github:
  client_id: ${GITHUB_CLIENT_ID}
  client_secret: ${GITHUB_CLIENT_SECRET}
  callback_url: http://localhost:8430/api/auth/callback

frontend:
  callback_url: http://localhost:5200/auth/callback
  login_url: http://localhost:5200/login

whitelist:
  users:
    - your-github-username
```

生成 `JWT_SECRET_KEY`：

```bash
uv run python scripts/generate_jwt_secret.py
```

GitHub OAuth App 本地配置：

| 字段 | 值 |
|---|---|
| Homepage URL | `http://localhost:5200` |
| Authorization callback URL | `http://localhost:8430/api/auth/callback` |

## 配置文件

根配置文件是 `config.yaml`，它会引用并合并 `config/` 下的子配置。

| 配置 | 路径 | 说明 |
|---|---|---|
| LLM | `config/llm_config.yaml` | 聊天、标题生成、压缩等 LLM 角色 |
| Memory | `config/memory_config.yaml` | 长短期记忆配置 |
| Server | `config/server_config.yaml` | host、port、CORS |
| Storage | `config/storage_config.yaml` | 聊天存储 |
| ASR | `config/asr_config.yaml` | 语音识别 |
| TTS | `config/tts_config.yaml` | 语音合成 |
| Auth | `config/auth.yaml` | OAuth、JWT、白名单 |

## 常用命令

```bash
# 运行测试
uv run pytest tests/ -q

# 运行认证相关测试
uv run pytest tests/auth tests/routes/test_auth.py tests/routes/test_chat_ws.py -q

# 代码检查
uv run ruff check src tests

# 类型检查
uv run python -m mypy src/ --ignore-missing-imports
```

## API 文档

后端启动后可访问：

- Swagger UI: `http://localhost:8430/docs`
- OpenAPI JSON: `http://localhost:8430/openapi.json`
- 项目接口文档: `../docs/后端API接口文档.md`

## 项目结构

```text
atri/
├── config/              # 子配置文件
├── data/                # 本地数据和上传资源
├── docs/                # 后端配置和开发文档
├── prompts/             # 角色人设
├── scripts/             # 辅助脚本
├── src/
│   ├── auth/            # OAuth、JWT、白名单和认证依赖
│   ├── middleware/      # HTTP 认证中间件
│   ├── routes/          # FastAPI 路由
│   ├── storage/         # 聊天、角色、Live2D 存储
│   ├── asr/             # ASR 服务
│   ├── tts/             # TTS 服务
│   ├── app.py           # FastAPI 应用工厂
│   └── main.py          # 服务启动入口
├── tests/               # 后端测试
├── config.yaml          # 根配置入口
├── pyproject.toml       # Python 项目配置
└── README.md
```

## 前端联调

前端仓库默认运行在 `http://localhost:5200`，Vite 代理会把 `/api` 和 `/ws` 转发到后端 `8430`。

认证开启时，确保 `config/auth.yaml` 中的前端回调地址使用同一个端口：

```yaml
frontend:
  callback_url: http://localhost:5200/auth/callback
  login_url: http://localhost:5200/login
```
