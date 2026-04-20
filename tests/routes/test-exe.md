# Phase 5 服务层测试验收文档

> **用途**: 本文档提供 Phase 5 FastAPI 服务层的测试验收指令。
> **参考**: docs/Phase5_执行规格.md §US-SRV-008

---

## 1. 快速自检（Mock 测试）

验证存储层和路由层的 mock 测试全部通过：

```bash
uv run pytest tests/storage/ tests/routes/ -v
```

**期望结果**:
- 所有测试通过（约 60+ 个测试）
- 无失败或错误

---

## 2. 全量测试（不含 Live）

运行完整测试套件（排除 live 测试）：

```bash
uv run pytest tests/ -q
```

**期望结果**:
- `259 passed, 4 deselected`（4 个 live 测试被排除）
- 无失败或错误

---

## 3. Live 端到端测试

运行真实 LLM 调用和 WebSocket 连接的集成测试：

```bash
uv run pytest tests/routes/test_live_server.py -m live -v -s
```

**前置条件**:
- `.env` 文件存在且包含有效的 API keys
- `config/llm_config.yaml` 配置正确
- 网络连接正常

**期望结果**:
- 5 个 live 测试全部通过：
  - `test_health_endpoint` - 健康检查
  - `test_characters_endpoint` - 角色列表包含 atri
  - `test_create_chat_with_llm_title` - LLM 生成标题
  - `test_websocket_three_rounds_conversation` - WebSocket 3 轮对话
  - `test_chat_update_and_delete` - 更新和删除聊天

---

## 4. 服务器启动冒烟测试

手动启动 FastAPI 服务器，验证启动日志：

```bash
uv run python -m src.main
```

**期望日志输出**（按顺序）:
```
INFO     | src.main:main:62 - atri starting
INFO     | src.main:main:65 - Dotenv loaded | path=... present=True
INFO     | src.main:main:68 - Config loaded | sections=[...]
INFO     | src.main:main:77 - LLM role resolved | role=chat | provider=... | model=...
INFO     | src.main:main:77 - LLM role resolved | role=l3_compress | provider=... | model=...
INFO     | src.main:main:77 - LLM role resolved | role=l4_compact | provider=... | model=...
INFO     | src.service_context:get_or_create_agent:166 - ChatAgent created | character=atri | user_id=main_demo | long_term=on
INFO     | src.main:main:93 - MemoryManager ready | mode=... | character=atri | long_term=on
INFO     | src.main:main:99 - ChatAgent ready | character=atri | persona=亚托莉 | long_term=on
INFO     | src.app:lifespan:55 - Starting FastAPI application
INFO     | src.app:lifespan:60 - Storage initialized: mode=json
INFO     | src.app:lifespan:64 - ServiceContext initialized
INFO     | src.main:main:115 - Server starting | host=0.0.0.0 | port=8000
INFO     | uvicorn.server:serve:76 - Started server process [...]
INFO     | uvicorn.lifespan.on:startup:47 - Waiting for application startup.
INFO     | uvicorn.lifespan.on:startup:61 - Application startup complete.
INFO     | uvicorn.server:_log_started_message:207 - Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**验证点**:
- ✅ 所有 3 个 LLM 角色成功解析
- ✅ ChatAgent 成功创建
- ✅ 出现 `Server starting | host=0.0.0.0 | port=8000`
- ✅ Uvicorn 成功启动

按 `Ctrl+C` 停止服务器。

---

## 5. WebSocket 手动测试（可选）

使用 `wscat` 工具手动测试 WebSocket 连接。

### 5.1 安装 wscat（如果未安装）

```bash
npm install -g wscat
```

### 5.2 启动服务器

在一个终端窗口中：

```bash
uv run python -m src.main
```

### 5.3 连接 WebSocket

在另一个终端窗口中：

```bash
wscat -c ws://localhost:8000/ws
```

### 5.4 发送测试消息

**测试 1: Ping/Pong**

发送：
```json
{"type": "ping"}
```

期望响应：
```json
{"type": "pong"}
```

**测试 2: 文本输入（需要先创建 chat）**

首先通过 REST API 创建聊天（在第三个终端）：

```bash
Invoke-RestMethod -Uri http://localhost:8430/api/chats -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"character_id":"atri","first_message":"你好"}'
```

记录返回的 `id` 字段（chat_id）。

然后在 wscat 中发送：
```json
{"type": "input:text", "data": {"text": "你好", "chat_id": "YOUR_CHAT_ID", "character_id": "atri"}}
```

期望响应：
- 多个 `{"type": "output:chat:chunk", "data": {"chunk": "...", ...}}`
- 最后一个 `{"type": "output:chat:complete", "data": {"full_reply": "...", ...}}`

**测试 3: 无效 JSON**

发送：
```
{invalid json
```

期望响应：
```json
{"type": "error", "data": {"message": "Invalid JSON format"}}
```

连接应该保持打开（不断开）。

**测试 4: 未知消息类型**

发送：
```json
{"type": "unknown:type"}
```

期望响应：
```json
{"type": "error", "data": {"message": "Unknown message type: unknown:type", ...}}
```

连接应该保持打开。

按 `Ctrl+C` 断开 wscat 连接。

---

## 6. 清理测试数据

### Bash / Git Bash / WSL

```bash
rm -rf data/chats/default/
rm -rf data/characters/*/sessions/*.json
```

### Windows CMD

```cmd
rmdir /s /q data\chats\default
del /q data\characters\*\sessions\*.json
```

### Windows PowerShell

```powershell
Remove-Item -Recurse -Force data/chats/default -ErrorAction SilentlyContinue
Remove-Item -Force data/characters/*/sessions/*.json -ErrorAction SilentlyContinue
```

---

## 7. 代码质量检查

### 7.1 类型检查

```bash
uv run python -m mypy src/ --ignore-missing-imports
```

**期望**: `Success: no issues found in 39 source files`

### 7.2 代码格式化和 Lint

```bash
uv run ruff format .
uv run ruff check . --fix
```

**期望**: `All checks passed!`

---

## 8. 验收清单

- [ ] 快速自检通过（60+ 个 mock 测试）
- [ ] 全量测试通过（259 passed, 4 deselected）
- [ ] Live 测试通过（5 个 live 测试）
- [ ] 服务器启动日志正确（包含 `Server starting` 行）
- [ ] WebSocket 手动测试通过（ping/pong + 文本输入 + 错误处理）
- [ ] mypy 类型检查通过
- [ ] ruff 格式化和 lint 通过
- [ ] 测试数据已清理

---

## 9. 故障排查

### 问题 1: Live 测试失败 - API Key 错误

**症状**: `LLMConnectionError` 或 `401 Unauthorized`

**解决**:
1. 检查 `.env` 文件是否存在
2. 验证 API key 是否有效
3. 确认 `config/llm_config.yaml` 中的 `base_url` 正确

### 问题 2: WebSocket 连接失败

**症状**: `Connection refused` 或 `404 Not Found`

**解决**:
1. 确认服务器已启动（`uv run python -m src.main`）
2. 检查端口 8000 是否被占用
3. 验证 WebSocket 路径为 `/ws`（不是 `/api/ws`）

### 问题 3: 测试数据残留

**症状**: 测试失败提示 "Chat already exists"

**解决**:
运行清理命令（见第 6 节）

### 问题 4: 端口被占用

**症状**: `Address already in use`

**解决**:

**Windows**:
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Linux/Mac**:
```bash
lsof -ti:8000 | xargs kill -9
```

或修改 `config/server_config.yaml` 中的 `port` 为其他值（如 8001）。

---

## 10. 完成标志

当以上所有验收清单项都勾选完成时，Phase 5 FastAPI 服务层实现完成，可以提交 PR。
