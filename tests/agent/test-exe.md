# Phase 4 ChatAgent — 测试执行指令

## 1. 快速自检（无网络）

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/agent/ -v
```

**期望**：
- `All checks passed!`
- `Success: no issues found in 28 source files`
- 所有 agent 层 mock 测试通过（persona loader 12 + chat_agent 18 + live_chat_agent 错误路径 1 = **31 passed**；live 10-round 用例因无 live 标识而 deselected）

## 2. 全量测试（无网络）

```bash
uv run pytest tests/ -q
```

**期望**：`206 passed, 4 deselected`
- 206 passed：Phase 1-3 累计 + Phase 4 新增（persona 12 + append_system_note 6 + chat_agent 18 + service_context 11 + live_chat_agent 错误路径 1）
- 4 deselected：2 个 Phase 2 live LLM + 1 个 Phase 3 live memory + **1 个 Phase 4 live ten-rounds**

## 3. Live 端到端测试（消耗真实 mem0 + LLM tokens）

### 前置 `.env`
```
MEM0_API_KEY=m0-...
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.2
```

### 命令
```bash
uv run pytest tests/agent/test_live_chat_agent.py -m live -v -s --basetemp=./tmp_live_agent
```

### 期望
- `1 passed in ~30-60s`（10 轮真实流式对话 + 2 次召回探针，依赖网络 / LLM 响应速度）
- `-s` 输出 12 段 log（10 轮真实对话 + 2 召回探针）：
  - `[round 1] USER: 我叫 Alice，最爱喝珍珠奶茶` → `[round 1]  AI : ...`（多种反应）
  - `[round 2]..[round 10]` 8 轮展开 + 1 轮话题转折
  - `[probe 1 - name]` 回复含 `"Alice"`
  - `[probe 2 - drink]` 回复含 `"珍珠奶茶"` 或 `"奶茶"`

### mem0 面板（app.mem0.ai）
筛选 `user_id=pytest_alice` / `agent_id=atri`：**10 轮未触发 L3**（`trigger_rounds=26`），所以**此阶段不应出现新 ADD 事实**。会话关闭后 `close_all` 会把 12 轮 recent_messages（10 对话 + 2 探针）作为 uncompressed tail 推到 mem0，届时面板才会出现 ADD 条目。

### 产物校验
```bash
# 路径含一个 pytest 生成的 subdir（参考 Phase 3 的实测：test_live_chat_agent_te0 / test_live_chat_agent_te1 ...）
ls tmp_live_agent/
```

- `tmp_live_agent/test_live_chat_agent_te0/atri/sessions/*.json` 应有**唯一一个**文件，共 21 个 JSON 对象：
  - `[0]` role=metadata，含 `session_id` / `character="atri"`
  - `[1..20]` 10 对 (human, ai)，按对话顺序
- `tmp_live_agent/test_live_chat_agent_te0/atri/short_term_memory.json`:
  - `total_rounds: 10`
  - `recent_messages` 长度 20
  - `active_blocks: []`
  - `meta_blocks: []`

**Note**：探针（round 11, 12）也会经过 `on_round_complete`，所以**测试断言在探针前做结构校验**；测试结束时 `total_rounds` 实际为 12。若你在 live 跑完后手动查 short_term，看到的是 12 不是 10，属正常。

## 4. main.py 冒烟

```bash
uv run python -m src.main
```

**期望末行**（按顺序）：
```
Config loaded | sections=['llm', 'memory']
LLM role resolved | role=chat provider=OpenAICompatibleLLM model=deepseek-ai/DeepSeek-V3.2
LLM role resolved | role=l3_compress provider=OpenAICompatibleLLM model=deepseek-ai/DeepSeek-V3.2
LLM role resolved | role=l4_compact provider=OpenAICompatibleLLM model=deepseek-ai/DeepSeek-V3.2
LongTermMemory ready | mode=sdk
ChatAgent created | character=atri | user_id=main_demo | long_term=on
MemoryManager ready | mode=sdk | character=atri | long_term=on
ChatAgent ready | character=atri | persona=亚托莉 | long_term=on
```

**Note**：`long_term=off` 若 mem0 构造失败（如 `MEM0_API_KEY` 未设置），属预期降级——`_safe_build_long_term` 返回 `None` 是一等支持状态。Windows CMD 下 `persona=亚托莉` 可能显示为 `persona=??????`，是 GBK 代码页的显示问题，日志文件里的 UTF-8 内容是正确的。

## 5. 错误路径离线验证（US-AGT-004 + US-AGT-007 集成）

已在 `test_live_chat_agent_llm_error`（默认纳入 pytest 全量跑）中覆盖。若要单独跑：

```bash
uv run pytest tests/agent/test_live_chat_agent.py::test_live_chat_agent_llm_error -v
```

**期望**：`1 passed in <1s`

该测试注入 stub LLM 抛 `LLMConnectionError`，验证真实 `ChatHistoryWriter` 落盘：
- chat_history 末尾有一条 `role=system` 行，content 以 `[LLM call failed: LLMConnectionError:` 开头
- 无 `role=ai` 行、无 `role=human` 行（本轮未完成，也不提交）
- `total_rounds == 0`，`recent_messages == []`

## 6. 清理

**Linux / macOS / Git Bash**
```bash
rm -rf tmp_live_agent/ data/characters/atri/
```

**Windows CMD**
```cmd
rmdir /s /q tmp_live_agent
rmdir /s /q data\characters\atri
```

**Windows PowerShell**
```powershell
Remove-Item -Recurse -Force tmp_live_agent, data\characters\atri
```
