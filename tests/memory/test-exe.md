# Phase 3 记忆系统 — 测试执行指令

## 1. 快速自检（无网络）

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/ -q
```

**期望**：
- `All checks passed!`
- `Success: no issues found in 25 source files`
- `155 passed, 3 deselected`

## 2. 仅跑 memory 模块

```bash
uv run pytest tests/memory/ -v
```

**期望**：`113 passed`

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
uv run pytest tests/memory/test_live_memory.py -m live -v -s --basetemp=./tmp_live_test
```

### 期望
- `1 passed in ~20s`
- `-s` 输出 2 条 recall probe：
  - `[recall probe - drink]` 回复含 `"奶茶"`
  - `[recall probe - pet]` 回复含 `"毛毛"`

### 产物校验
```bash
cat tmp_live_test/test_live_memory_manager_full_0/short_term_memory.json
```
- `total_rounds: 26`
- `active_blocks` 长度 1，`covers_rounds: [1, 20]`
- `active_blocks[0].summary` markdown，含 5 个主题（珍珠奶茶/毛毛/钢琴/杭州/飞机），无 `<analysis>` 残留
- `recent_messages` 12 条

### mem0 面板（app.mem0.ai）
筛选 `user_id=pytest_alice` / `agent_id=atri_live`：应有 **5 条 ADD 事实**。

## 4. main.py 冒烟

```bash
uv run python -m src.main
```

**期望末行**：
```
MemoryManager ready | mode=local_deploy | character=atri | long_term=on|off
```

## 5. 清理

**Linux / macOS / Git Bash**
```bash
rm -rf tmp_live_test/ data/characters/atri/
```

**Windows CMD**
```cmd
rmdir /s /q tmp_live_test
rmdir /s /q data\characters\atri
```

**Windows PowerShell**
```powershell
Remove-Item -Recurse -Force tmp_live_test, data\characters\atri
```
