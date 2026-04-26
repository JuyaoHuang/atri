# Emotion Robot

<h3 align="center">以 ATRI 为核心的可部署 AI 角色伴侣项目</h3>

Emotion Robot 是一个面向本地部署和个人私有化使用的 AI 角色伴侣系统。项目把后端服务、前端交互界面、角色配置、记忆系统、语音能力和 Live2D 表现层组织在一起，目标是构建一个可持续扩展的虚拟角色对话平台。

本仓库 `atri` 是项目的母仓库和后端主仓库。后续前端仓库 `atri-webui` 会作为 Git 子模块挂载到本仓库中；在当前开发工作区中，前端仓库暂时位于同级目录 `../atri-webui`。

## 项目状态

当前实现覆盖到 Phase 11：

- 后端 FastAPI 服务，默认地址 `http://localhost:8430`
- Vue 3 前端应用，默认地址 `http://localhost:5200`
- WebSocket 流式对话，默认端点 `ws://localhost:8430/ws`
- GitHub OAuth + JWT 可选认证系统
- 多用户聊天数据隔离，认证关闭时使用本地 `default` 用户
- 角色创建、编辑、删除和头像上传
- 聊天会话创建、历史记录、标题更新和删除
- ASR / TTS 提供商配置、切换和健康检查
- Live2D 模型上传、资源托管和表情列表接口

## 功能亮点

- 可本地运行：认证可以关闭，适合单机部署和本地开发。
- 可公网部署：认证开启后通过 GitHub OAuth 登录，并使用白名单限制用户。
- 角色可定制：支持角色人设、问候语、简介和头像管理。
- 对话可持久化：聊天历史保存在后端存储中，认证开启时按用户隔离。
- 流式交互：通过 WebSocket 接收 LLM 分片输出，前端实时展示回复。
- 语音链路可扩展：ASR 和 TTS 通过后端统一接口管理提供商。
- Live2D 可接入：后端托管模型资源，前端负责加载和交互展示。
- 配置分层：`config.yaml` 引用各子配置，便于模块化调整。

## 系统架构

```text
atri/                         # Emotion Robot 母仓库
├── src/                      # FastAPI、Agent、记忆、ASR、TTS、存储、认证
├── config/                   # LLM、记忆、服务、存储、ASR、TTS、认证配置
├── prompts/                  # 角色人设
├── data/                     # 本地数据、头像、Live2D 上传资源
├── docs/                     # 配置说明、设计文档、开发记录
├── tests/                    # 后端测试
└── atri-webui/               # 前端子模块；当前工作区尚未挂载时位于 ../atri-webui
    ├── src/                  # Vue 页面、组件、状态、API 客户端
    └── docs/                 # 前端开发和使用文档
```

数据流：

```text
用户浏览器
  -> atri-webui: Vue UI、路由守卫、LocalStorage token
  -> atri: HTTP API、WebSocket、OAuth 回调、聊天存储
  -> LLM / ASR / TTS / Memory providers
```

## 技术栈

后端：

| 模块 | 技术 |
|---|---|
| Web 服务 | FastAPI + Uvicorn |
| 配置 | YAML + python-dotenv |
| LLM | OpenAI 兼容接口 |
| 认证 | GitHub OAuth + JWT |
| 存储 | 本地 JSON 存储起步，保留扩展空间 |
| 测试 | pytest、ruff、mypy |

前端：

| 模块 | 技术 |
|---|---|
| 框架 | Vue 3 + TypeScript |
| 构建 | Vite |
| 状态 | Pinia |
| 样式 | UnoCSS |
| 通信 | Axios + WebSocket |
| Live2D | PixiJS + pixi-live2d-display |

## 快速开始

### 1. 准备环境

需要：

- Python `>= 3.11`
- `uv`
- Node.js `>= 18`
- npm `>= 9`
- 一个 OpenAI 兼容 LLM 服务

### 2. 启动后端

在 `atri` 仓库中安装依赖：

```bash
uv sync
```

创建 `.env`：

```powershell
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

验证：

```bash
curl http://localhost:8430/health
```

### 3. 启动前端

当前开发工作区中，前端仓库位于 `../atri-webui`：

```bash
cd ../atri-webui
npm install
npm run dev
```

访问：

```text
http://localhost:5200
```

当前 Vite 开发端口固定为 `5200`，并代理：

| 前端路径 | 代理目标 |
|---|---|
| `/api/*` | `http://localhost:8430/api/*` |
| `/ws` | `ws://localhost:8430/ws` |

后续前端作为子模块挂载后，启动路径会变为：

```bash
cd atri-webui
npm install
npm run dev
```

## 认证系统

认证配置位于 `config/auth.yaml`。

本地部署或个人开发可以关闭认证：

```yaml
enabled: false
```

关闭认证时：

- 前端直接进入主页面。
- 后端使用 `default` 用户。
- HTTP 和 WebSocket 都不需要 token。

公网部署建议开启认证：

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

认证开启后：

- 未登录用户访问前端主页面会跳转到 `/login`。
- 登录成功后，后端回调到 `/auth/callback` 并返回 JWT。
- HTTP 请求使用 `Authorization: Bearer <JWT_TOKEN>`。
- WebSocket 使用 `ws://localhost:8430/ws?token=<JWT_TOKEN>`。

## 配置说明

根配置文件是 `config.yaml`，它引用并合并 `config/` 下的子配置。

| 配置 | 路径 | 说明 |
|---|---|---|
| LLM | `config/llm_config.yaml` | 聊天、标题生成、压缩等 LLM 角色 |
| Memory | `config/memory_config.yaml` | 长短期记忆配置 |
| Server | `config/server_config.yaml` | host、port、CORS |
| Storage | `config/storage_config.yaml` | 聊天存储 |
| ASR | `config/asr_config.yaml` | 语音识别 |
| TTS | `config/tts_config.yaml` | 语音合成 |
| Auth | `config/auth.yaml` | OAuth、JWT、白名单 |

环境变量模板在 `.env.example`。不要提交真实密钥。

## 常用命令

后端：

```bash
# 运行全部默认测试
uv run pytest tests/ -q

# 运行认证相关测试
uv run pytest tests/auth tests/routes/test_auth.py tests/routes/test_chat_ws.py -q

# 代码检查
uv run ruff check src tests

# 类型检查
uv run python -m mypy src/ --ignore-missing-imports
```

前端：

```bash
# 开发服务器
npm run dev

# 生产构建
npm run build

# 类型检查
npm run type-check

# ESLint 检查并自动修复
npm run lint
```

## 文档入口

| 文档 | 路径 |
|---|---|
| 后端 API 接口 | `docs/developments/module-design/后端API接口文档.md` |
| 认证系统使用指南 | `docs/configs/认证系统使用指南.md` |
| ASR 配置说明 | `docs/configs/ASR配置说明.md` |
| TTS 配置说明 | `docs/configs/TTS配置说明.md` |
| 角色创建指南 | `docs/configs/角色创建指南.md` |
| 项目架构设计 | `docs/developments/项目架构设计.md` |
| 后端执行准则 | `执行准则.md` |
| 前端执行准则 | 当前 `../atri-webui/执行准则.md`，子模块后为 `atri-webui/执行准则.md` |
| 前端 README | 当前 `../atri-webui/README.md`，子模块后为 `atri-webui/README.md` |

后端启动后也可以访问：

- Swagger UI: `http://localhost:8430/docs`
- OpenAPI JSON: `http://localhost:8430/openapi.json`

## 部署要点

部署到服务器时，至少需要检查：

- `config/server_config.yaml` 的 host、port 和 CORS。
- `.env` 中的 LLM、TTS、Memory、JWT、GitHub OAuth 密钥。
- `config/auth.yaml` 中的 `enabled`、OAuth 回调地址、前端地址和白名单。
- 前端生产环境的 `VITE_API_BASE_URL` 和 `VITE_WS_URL`。
- 如果不是 localhost，浏览器麦克风等能力通常需要 HTTPS。

## License

MIT
