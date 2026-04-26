## 环境要求

- Python `>= 3.11`
- [uv](https://docs.astral.sh/uv/) 用于管理 Python 依赖
- Node.js `>= 18`
- npm `>= 9`

推荐使用两个终端分别启动后端和前端。

## 1. 一键安装

> 如需 docker部署，提供 [Docker 部署指南](./configs/'Docker部署指南.md)

如果你已经克隆了 `atri` 主仓库，在主仓库根目录运行：

```powershell
.\install.bat --skip-clone
```

Linux / macOS：

```bash
bash install.sh --skip-clone
```

脚本会自动完成：

- 初始化并拉取 `frontend` 子模块
- 从 `.env.example` 创建 `.env`
- 使用清华 PyPI 镜像安装后端依赖
- 使用 npm 国内镜像安装前端依赖

默认镜像：

| 依赖 | 默认镜像 |
|---|---|
| Python / PyPI | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| npm | `https://registry.npmmirror.com` |

阿里云 PyPI 备用镜像：

```text
https://mirrors.aliyun.com/pypi/simple/
```

如果你在空目录里运行脚本，可以让它先克隆仓库：

```powershell
.\install.bat --repo-url https://github.com/JuyaoHuang/atri.git --target-dir atri
```

Linux / macOS：

```bash
bash install.sh --repo-url https://github.com/JuyaoHuang/atri.git --target-dir atri
```

如需更换镜像：

```powershell
.\install.bat --pypi-index https://pypi.tuna.tsinghua.edu.cn/simple --npm-registry https://registry.npmmirror.com
```

使用阿里云 PyPI 镜像：

```powershell
.\install.bat --pypi-index https://mirrors.aliyun.com/pypi/simple/ --npm-registry https://registry.npmmirror.com
```

安装完成后，编辑 `.env` 填写 `SILICONFLOW_API_KEY` 和 `COMPRESS_API_KEY`，再分别启动后端和前端。

## 2. 手动获取代码

首次克隆主仓库时，推荐直接拉取子模块：

```powershell
git clone --recurse-submodules https://github.com/JuyaoHuang/atri.git
cd atri
```

如果已经克隆过主仓库，再初始化前端子模块：

```powershell
git submodule update --init --recursive
```

以后更新前端子模块：

```powershell
git submodule update --remote frontend
```

当前前端子模块配置：

| 项 | 值 |
|---|---|
| 子模块路径 | `frontend` |
| 子模块仓库 | `https://github.com/JuyaoHuang/atri-webui.git` |
| 跟踪分支 | `main` |

## 3. 启动后端

确认当前位于 `atri` 主仓库根目录。如果你打开了新的终端，先进入主仓库：

```powershell
cd path\to\atri
```

安装依赖：

```powershell
uv sync
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填写主聊天模型和压缩模型使用的密钥：

```env
SILICONFLOW_API_KEY=sk-xxxx
COMPRESS_API_KEY=sk-xxxx
```

默认配置使用 OpenAI 兼容接口，模型和接口地址在 `config/llm_config.yaml` 中配置。你可以把 `base_url` 和 `model` 改成自己的服务商，例如 SiliconFlow、DeepSeek 或其他 OpenAI-compatible API。

启动后端：

```powershell
uv run python -m src.main
```

默认地址：

| 服务 | 地址 |
|---|---|
| HTTP API | `http://localhost:8430` |
| WebSocket | `ws://localhost:8430/ws` |
| Swagger UI | `http://localhost:8430/docs` |
| OpenAPI JSON | `http://localhost:8430/openapi.json` |

后端能正常启动时，日志里会出现 `Server starting | host=0.0.0.0 | port=8430`。

## 4. 启动前端

打开第二个终端，进入前端目录：

```powershell
cd path\to\atri\frontend
```

安装依赖：

```powershell
npm install
```

确认 `.env.development` 指向本地后端：

```env
VITE_API_BASE_URL=http://localhost:8430
VITE_WS_URL=ws://localhost:8430/ws
```

启动前端：

```powershell
npm run dev
```

访问：

```text
http://localhost:5200
```

## 5. 最小配置说明

ATRI 使用根配置 `config.yaml` 作为入口，再加载 `config/` 下的子配置：

| 配置文件 | 作用 |
|---|---|
| `config/llm_config.yaml` | 聊天模型、L3 压缩模型、L4 压缩模型、标题生成模型 |
| `config/memory_config.yaml` | 三层记忆压缩、mem0 长期记忆、向量检索参数 |
| `config/server_config.yaml` | 后端监听地址、端口、CORS、WebSocket 心跳 |
| `config/storage_config.yaml` | 聊天记录存储方式，默认本地 JSON |
| `config/asr_config.yaml` | 语音识别提供商 |
| `config/tts_config.yaml` | 语音合成提供商 |
| `config/auth.yaml` | GitHub OAuth、JWT、白名单和认证开关 |

本地开发时通常只需要改：

```text
.env
config/llm_config.yaml
config/memory_config.yaml
config/auth.yaml
```

## 6. 记忆系统配置

ATRI 的核心是记忆存储。它把对话记忆分成三层：

| 层级 | 触发时机 | 作用 |
|---|---|---|
| L1 Snip | 每轮对话 | 规则清洗填充词、重复输入和超长输入 |
| L3 Collapse | 每 26 轮 | 把较早 20 轮压缩成事件级摘要 |
| L4 Super-Compact | 每 4 个 L3 block | 把多个事件摘要提炼成长期画像和模式 |

这些参数在 `config/memory_config.yaml` 中配置：

```yaml
short_term:
  collapse:
    trigger_rounds: 26
    compress_rounds: 20
    keep_recent_rounds: 6
  super_compact:
    trigger_blocks: 4
```

长期记忆由 mem0 提供。默认模式是 `sdk`：

```yaml
mem0:
  mode: sdk
  sdk:
    api_key: ${MEM0_API_KEY}
```

如果使用 mem0 SaaS，需要在 `.env` 中填写：

```env
MEM0_API_KEY=m0-xxxx
```

如果暂时不想接入云端 mem0，可以把 `config/memory_config.yaml` 中的模式改为本地部署：

```yaml
mem0:
  mode: local_deploy
```

本地部署默认使用 `./data/qdrant` 作为嵌入式向量存储，并复用 `.env` 中的 `SILICONFLOW_API_KEY` 调用 embedding 和事实抽取模型。

如果云端部署时不想使用 mem0 SaaS，可以继续使用 `mem0.local_deploy`，并把向量存储切换到 Neon PostgreSQL + pgvector：

```yaml
mem0:
  mode: local_deploy
  local_deploy:
    vector_store:
      provider: pgvector
      providers:
        qdrant:
          config:
            path: ./data/qdrant
        pgvector:
          config:
            connection_string: ${DB_MEMORY_URL}
            collection_name: atri_memories
            embedding_model_dims: 1024
            hnsw: true
            diskann: false
            minconn: 1
            maxconn: 5
```

`provider` 表示当前启用的向量库；`providers` 保留多套配置。默认使用 `qdrant` 时，`pgvector` 分支里的 `${DB_MEMORY_URL}` 不会被强制校验。切换到 `pgvector` 前，需要在 Neon 数据库中启用 `vector` 扩展，并确保 `.env` 已配置 `DB_MEMORY_URL`。

## 7. 认证配置

本地开发默认关闭认证：

```yaml
# config/auth.yaml
enabled: false
```

关闭认证时，前端会直接进入主页面，后端使用默认用户身份，适合单机部署和本地调试。

公网部署时建议启用认证：

```yaml
enabled: true
```

启用认证前，需要在 `.env` 中配置：

```env
JWT_SECRET_KEY=replace-with-a-long-random-secret
GITHUB_CLIENT_ID=Iv1.xxxxx
GITHUB_CLIENT_SECRET=github_oauth_client_secret
```

生成 `JWT_SECRET_KEY`：

```powershell
uv run python scripts/generate_jwt_secret.py
```

GitHub OAuth App 本地开发推荐填写：

| GitHub OAuth 字段 | 值 |
|---|---|
| Homepage URL | `http://localhost:5200` |
| Authorization callback URL | `http://localhost:8430/api/auth/callback` |

`config/auth.yaml` 中的前端回调也要和端口一致：

```yaml
frontend:
  callback_url: http://localhost:5200/auth/callback
  login_url: http://localhost:5200/login
```

白名单控制哪些 GitHub 用户允许登录：

```yaml
whitelist:
  users:
    - your-github-username
```

JWT 默认有效期是 7 天：

```yaml
jwt:
  expire_days: 7
```

## 8. 常用开发命令

后端：

```powershell
cd path\to\atri

# 启动服务
uv run python -m src.main

# 运行测试
uv run pytest

# Ruff 检查
uv run ruff check src tests

# Mypy 检查
uv run python -m mypy src/ --ignore-missing-imports
```

前端：

```powershell
cd path\to\atri\frontend

# 启动开发服务器
npm run dev

# 类型检查
npm run type-check

# 生产构建
npm run build
```

子模块常用命令：

```powershell
cd path\to\atri

# 查看子模块状态
git submodule status

# 初始化或恢复子模块内容
git submodule update --init --recursive

# 拉取 frontend 跟踪分支的最新提交
git submodule update --remote frontend
```

如果你在 `frontend/` 中提交了前端改动，需要先在子模块仓库提交并推送，再回到 `atri` 提交子模块指针变化：

```powershell
cd path\to\atri\frontend
git add .
git commit --no-gpg-sign -m "your frontend change"
git push

cd ..
git add frontend
git commit --no-gpg-sign -m "chore: update frontend submodule"
```

## 9. 数据目录

默认运行时数据写入主仓库的 `data/`：

| 目录 | 内容 |
|---|---|
| `data/chats` | 聊天会话和历史消息 |
| `data/characters` | 角色配置、头像等资源 |
| `data/live2d` | Live2D 模型资源 |
| `data/qdrant` | 本地 mem0 向量存储 |

`chat_history` 是记忆系统的事实来源。短期记忆文件损坏时，可以基于聊天历史重建。

## 10. 常见问题

### frontend 目录为空

说明前端子模块还没有初始化。在 `atri` 目录执行：

```powershell
git submodule update --init --recursive
```

### frontend 不是最新版本

在 `atri` 目录执行：

```powershell
git submodule update --remote frontend
```

如果 `atri` 出现 `modified: frontend`，表示子模块指针变化了。确认前端版本正确后，需要在主仓库提交这个指针变化。

### 后端启动后立刻退出

通常是 `.env` 中缺少 `SILICONFLOW_API_KEY` 或 `COMPRESS_API_KEY`，或者 `config/llm_config.yaml` 中的模型地址不可用。先检查后端日志中的 `LLM role failed`。

### 前端打不开或请求失败

确认后端正在运行，并检查 `frontend/.env.development`：

```env
VITE_API_BASE_URL=http://localhost:8430
VITE_WS_URL=ws://localhost:8430/ws
```

修改后需要重启 `npm run dev`。

### WebSocket 返回 403

如果认证已启用，WebSocket 需要携带有效 JWT。请先在前端完成登录。  
如果是本地单机使用，可以把 `config/auth.yaml` 改为：

```yaml
enabled: false
```

然后重启后端。

### 登录后仍然被跳回登录页

常见原因：

- `JWT_SECRET_KEY` 改过，但浏览器还保存着旧 token
- GitHub 用户名不在 `whitelist.users`
- GitHub OAuth callback URL 和 `config/auth.yaml` 不一致
- 前端 `VITE_API_BASE_URL` 指向了错误后端

可以先清理浏览器 LocalStorage 中的 `atri_auth_token`，再重新登录。

### Swagger 可以访问，但聊天无回复

检查：

- `.env` 中的 LLM API key 是否有效
- `config/llm_config.yaml` 的 `base_url` 是否包含 `/v1`
- 模型名是否被当前服务商支持
- 压缩模型 `COMPRESS_API_KEY` 是否也已填写

## 11. 下一步

- 阅读 `docs/configs/认证系统使用指南.md` 配置公网登录。
- 阅读 `docs/configs/ASR配置说明.md` 和 `docs/configs/TTS配置说明.md` 启用语音链路。
- 阅读 `docs/configs/角色创建指南.md` 添加自己的角色、头像和问候语。
- 通过 `http://localhost:8430/docs` 查看后端接口。
