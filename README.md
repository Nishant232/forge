# ⚒ Forge — Local AI Coding Agent

> **Fully private. Zero cloud. Runs entirely on your machine.**

**Forge** is a local AI coding agent powered by [Ollama](https://ollama.ai).
Give it a task in plain English and it reads, edits, runs, and tests your code
autonomously through a ReAct (Reason + Act) loop — just like Claude Code,
but with no API keys, no subscriptions, and no data ever leaving your network.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Ollama](https://img.shields.io/badge/requires-Ollama-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

1. [How It Works — The Big Picture](#1-how-it-works--the-big-picture)
2. [Architecture](#2-architecture)
3. [The ReAct Loop in Detail](#3-the-react-loop-in-detail)
4. [Tools Reference](#4-tools-reference)
5. [Memory and Persistence](#5-memory-and-persistence)
6. [Safety and Trust Modes](#6-safety-and-trust-modes)
7. [Post-Edit Verification](#7-post-edit-verification)
8. [Context Management](#8-context-management)
9. [Custom Instructions — agentX.md](#9-custom-instructions--agentxmd)
10. [Sub-Agents](#10-sub-agents)
11. [Installation](#11-installation)
12. [Configuration](#12-configuration)
13. [CLI Reference](#13-cli-reference)
14. [Slash Commands](#14-slash-commands)
15. [Project Layout](#15-project-layout)
16. [Known Limitations](#16-known-limitations)

---

## 1. How It Works — The Big Picture

```
You type:  "add input validation to the signup endpoint"
                         │
                         ▼
            ┌────────────────────────┐
            │   System Prompt        │  ← trust mode, project root,
            │   + agentX.md          │    custom instructions,
            │   + past episodes      │    similar past tasks
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   Ollama LLM           │  ← qwen2.5:14b (default)
            │   (ChatOllama)         │    streaming, tool-call aware
            └────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │ tool calls?          │
         YES  │                      │ NO → verification → done
              ▼                      ▼
  ┌─────────────────────┐   ┌──────────────────────┐
  │  Parallel dispatch  │   │  Post-edit verifier   │
  │  (asyncio.gather)   │   │  ruff / mypy / pytest │
  └─────────────────────┘   └──────────────────────┘
              │                      │
    tool results back         failures? loop back
    into message history       with feedback
              │
         loop again
```

A single **turn** (one user message → one agent reply) runs the entire loop,
potentially making dozens of tool calls across many iterations before returning
a final answer.

---

## 2. Architecture

```
src/agent/
├── cli.py                  ← Entry points: `agentX` and `agent` commands
├── server.py               ← Optional FastAPI REST server
│
├── core/
│   ├── loop.py             ← THE MAIN ENGINE: run_turn(), ReAct loop
│   └── history.py          ← SQLite-backed conversation persistence
│
├── llm/
│   ├── chat.py             ← ChatOllama wrapper (chat_model, with_tools, structured)
│   ├── embed.py            ← Ollama embeddings via /api/embed
│   ├── prompts.py          ← Prompt templates
│   └── schemas.py          ← Pydantic schemas for structured LLM output
│
├── tools/
│   ├── registry.py         ← all_tools() list + dispatch()
│   ├── files.py            ← read_file, write_file, edit_file, edit_file_multi,
│   │                          list_dir, find_files
│   ├── shell.py            ← run_shell (sandboxed)
│   ├── search.py           ← search_code (ripgrep)
│   ├── tests.py            ← run_tests (pytest)
│   ├── git.py              ← git_status/diff/add/commit/log/checkpoint/rollback
│   ├── web.py              ← fetch_url, search_web (yolo mode only)
│   ├── subtask.py          ← run_subtask (spawn a sub-agent)
│   └── todo.py             ← todo_write, todo_read (per-session checklist)
│
├── memory/
│   ├── episodic.py         ← Past task store with cosine-similarity search
│   ├── project.py          ← Per-project facts (language, framework, conventions)
│   ├── working.py          ← In-flight context assembly
│   └── consolidator.py     ← Background fact extraction
│
├── safety/
│   ├── policy.py           ← Trust modes, allowlists, path sandboxing
│   └── approval.py         ← Interactive approval prompts
│
├── verify/
│   └── runner.py           ← Post-edit: ruff, mypy, pytest / eslint, tsc / cargo
│
├── context/
│   ├── assembler.py        ← Token-budget-aware context assembly
│   └── budgeter.py         ← Token counting
│
└── code_index/             ← Optional: vector + symbol index (requires [index] extra)
    ├── chunker.py
    ├── parser.py
    ├── search.py
    ├── symbols.py
    ├── vectors.py
    └── watcher.py
```

The **active path** for every task is:

```
cli.py  →  core/loop.py:run_turn()  →  tools/*  →  llm/chat.py
```

---

## 3. The ReAct Loop in Detail

`core/loop.py` is the heart of agentX. The full flow of `run_turn()`:

```python
async def run_turn(user_message, *, workspace, session_id, history, trust,
                   on_content, on_content_token, on_tool_start, on_tool_end)
```

### Step-by-step

```
1. Set env vars
   AGENT_PROJECT_ROOT, AGENT_TRUST_MODE, AGENT_SESSION_ID

2. Retrieve episodic memory
   embed(user_message) → cosine search → top-2 similar past tasks
   (silently skipped if Ollama embed model not available)

3. Build system prompt
   project root + trust mode + project conventions + agentX.md + past episodes

4. Load + compact history
   history.load(session_id)
   if len > 60 messages → _maybe_compact() (LLM summarises old messages)
   if len > 80 messages → hard trim to last 80

5. Build message list
   [SystemMessage, ...history, HumanMessage(user_message)]

6. Bind tools to model
   ChatOllama.bind_tools(all_tools())   ← 21 tools

7. ReAct loop (max 40 iterations)
   ┌─────────────────────────────────────────────┐
   │  a. _call_model() — stream tokens via        │
   │     model.astream(), accumulate AIMessage    │
   │                                              │
   │  b. If no tool calls:                        │
   │     → run post-edit verifier on changed files│
   │     → if verifier fails: inject feedback,    │
   │       loop back (max 2 retries)              │
   │     → else: save to history, break           │
   │                                              │
   │  c. If tool calls present:                   │
   │     → auto-checkpoint before first write     │
   │     → _dispatch_parallel() via asyncio.gather│
   │       (all tool calls in one response run    │
   │        concurrently in thread pool)          │
   │     → truncate outputs > 8,000 chars         │
   │     → append ToolMessages to history         │
   │     → loop                                   │
   └─────────────────────────────────────────────┘

8. Save episode to episodic memory

9. Return TurnResult(text, files_changed, tool_calls_made, iterations)
```

### Key constants (loop.py)

| Constant | Value | Purpose |
|---|---|---|
| `_MAX_ITERATIONS` | 40 | Hard cap on reasoning steps per turn |
| `_COMPACT_THRESHOLD` | 60 | Message count that triggers auto-compact |
| `_MAX_HISTORY_MESSAGES` | 80 | Hard trim if compact fails |
| `_MAX_TOOL_OUTPUT` | 8,000 chars | Tool output truncation limit |
| `_MAX_VERIFIER_RETRIES` | 2 | Max loop-backs for failed verification |

### Streaming

The model is called with `model.astream()`. Each text chunk is forwarded immediately
to the terminal via `on_content_token`. If streaming fails (some Ollama models don't
support it), the loop falls back to `model.ainvoke()` automatically.

```python
async for chunk in model.astream(messages):
    if isinstance(chunk.content, str) and chunk.content:
        on_token(chunk.content)   # printed to terminal live
    full = chunk if full is None else full + chunk
```

### Parallel tool dispatch

When the LLM returns multiple tool calls in one response, they all run at the same time:

```python
# All tool calls dispatched concurrently
results = await asyncio.gather(*[_one(tc) for tc in tool_calls])

# Each tool runs in its own thread (not blocking the event loop)
output = await asyncio.to_thread(tool.invoke, args)
```

---

## 4. Tools Reference

agentX has 21 tools registered in `tools/registry.py`. The LLM chooses which to call.

### File Operations (`tools/files.py`)

| Tool | Signature | Description |
|---|---|---|
| `read_file` | `(path, start_line=1, end_line=None)` | Read whole file or a line range. Range output includes line numbers for precise editing. |
| `write_file` | `(path, content)` | Create or overwrite a file. Creates parent dirs automatically. |
| `edit_file` | `(path, old_string, new_string)` | Replace exactly one occurrence of a string. Fails if 0 or 2+ matches (forces precision). |
| `edit_file_multi` | `(path, old_strings, new_strings)` | Apply multiple edits to one file in a single call. More efficient than chaining `edit_file`. |
| `list_dir` | `(path)` | List directory entries with `[F]`/`[D]` prefix. |
| `find_files` | `(pattern, path=".")` | Glob search. `**/*.py` finds all Python files recursively. Max 500 results. |

All file tools enforce **path sandboxing**: every path is resolved and verified to be
inside `AGENT_PROJECT_ROOT`. Attempts to access `../` or absolute paths outside the
project raise `PermissionError`.

### Shell & Search

| Tool | Signature | Description |
|---|---|---|
| `run_shell` | `(command, timeout=120)` | Run any shell command in the project root. Subject to trust mode and allowlist. |
| `search_code` | `(pattern, file_glob="", max_results=50)` | Full-text search via `rg` (ripgrep). Returns `path:line:content` format. |
| `run_tests` | `(test_path="", extra_args="")` | Run pytest. Empty `test_path` runs the whole suite. Timeout 300s. |

### Git (`tools/git.py`)

| Tool | Description |
|---|---|
| `git_status` | Short working-tree status |
| `git_diff` | Show unstaged changes (optionally for one path) |
| `git_add` | Stage a file or directory |
| `git_commit` | Commit staged changes. Blocked on protected branches (main, master, develop, production, release). |
| `git_log` | Last N commits as one-line summary |
| `git_checkpoint` | Create a lightweight git tag as a rollback point (`agent-cp-<label>-<ts>`) |
| `git_rollback` | Reset working tree to a previous checkpoint tag |

**Auto-checkpoint**: before the agent's first write operation in any turn, `run_turn`
automatically creates a git tag `agent-cp-auto-<timestamp>` so you can always undo.

### Web (`tools/web.py`) — trust=yolo only

| Tool | Description |
|---|---|
| `fetch_url` | HTTP GET a URL, returns text (truncated at 50,000 chars) |
| `search_web` | DuckDuckGo search, returns title + URL + snippet for top N results |

These tools are completely blocked unless `AGENT_TRUST_MODE=yolo`.
`search_web` also requires the `[search]` extra: `pip install 'local-coding-agent[search]'`.

### Agent & Task Management

| Tool | Description |
|---|---|
| `run_subtask` | Spawn a fresh sub-agent with its own context for a focused sub-task |
| `todo_write` | Set the session checklist. Prefix items with `[done] ` to mark complete. |
| `todo_read` | Read the current checklist with `○`/`✓` markers |

---

## 5. Memory and Persistence

agentX stores all state in `.agent/` inside your project directory.

```
your-project/
└── .agent/
    ├── state.db        ← SQLite: sessions, messages, episodes, project facts
    ├── config.toml     ← Trust mode, shell allowlist, settings
    ├── todos/
    │   └── <session-id[:8]>.json   ← Per-session todo list
    └── checkpoints/
        └── index.txt   ← Git checkpoint tag names
```

### Conversation History (`core/history.py`)

Every message (human, AI, tool call, tool result) is stored in SQLite and replayed
on `resume`. Messages are persisted in batched transactions for efficiency.

```sql
CREATE TABLE conv_sessions (
    id TEXT PRIMARY KEY, workspace TEXT, title TEXT,
    created_at TEXT, updated_at TEXT
);
CREATE TABLE conv_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, role TEXT, content TEXT,
    tool_calls TEXT, tool_call_id TEXT, tool_name TEXT,
    created_at TEXT
);
```

Messages are serialized per role:
- `human` → `{role, content}`
- `ai` → `{role, content, tool_calls: JSON}`
- `tool` → `{role, content, tool_call_id, tool_name}`

### Episodic Memory (`memory/episodic.py`)

After every turn, agentX saves an **episode** recording:
- the user's request
- which files were changed
- how many tool calls were made
- outcome (success/partial/failure)

At the start of each turn, the top-2 most **semantically similar** past episodes are
retrieved (via cosine similarity on Ollama embeddings) and injected into the system
prompt. This lets the agent learn from how it solved similar tasks before.

```python
# Retrieval at turn start
vecs = await embed_texts([user_message])   # Ollama nomic-embed-text
eps = store.search_semantic(vecs[0], k=2)
# Injected as: "## Similar past tasks\n  [success] add login endpoint..."
```

Cosine similarity is computed in-process — no external vector database needed.

### Project Facts (`memory/project.py`)

agentX detects and stores project-level facts (language, framework, test runner,
code conventions) in `state.db`. These are injected into every system prompt under
`## Project conventions`.

---

## 6. Safety and Trust Modes

Trust mode controls what the agent is allowed to do. Set it with:

```bash
agent config trust_mode trusted   # persisted in .agent/config.toml
agentX --trust readonly           # or per-session flag
```

### Modes

| Mode | File writes | Shell writes | Network tools |
|---|---|---|---|
| `readonly` | Blocked | Blocked | Blocked |
| `trusted` | Allowed | Allowed | Blocked (default) |
| `yolo` | Allowed | Allowed | Allowed |

### Permanent denies (all modes, cannot be overridden)

These commands are always blocked regardless of trust mode or allowlist:

```
rm -rf   dd   mkfs   shred   fdisk   parted   wipefs   writes to /dev/*
```

### Shell allowlist

You can allow specific commands that would otherwise be blocked by adding them to
`.agent/config.toml`:

```toml
shell_allowlist = [
    "npm run *",
    "pytest *",
    "python -m pytest *",
    "cargo test",
]
```

Glob patterns (`*`, `?`) are supported. Allowlisted commands skip trust-mode
write/network restrictions but **cannot** override the permanent deny list.

### Path sandboxing

All file tools resolve the target path and verify it lives inside `AGENT_PROJECT_ROOT`.
Any path that escapes the project root (via `../` traversal or absolute paths) raises
`PermissionError` before the file is touched.

---

## 7. Post-Edit Verification

After the agent finishes writing code (no more tool calls), agentX automatically
runs your project's linter, type checker, and tests on the changed files.
If any check fails, the failure output is fed back to the agent as a new message
and the loop continues — up to 2 times.

```
agent edits files
       │
       ▼
run_verifier(changed_files, workspace)
       │
 ┌─────┴─────┐
PASS        FAIL
 │           │
done    inject feedback:
        "Verification failed...
         ruff: line 42: undefined name 'x'
         Fix the issues before finishing."
             │
             ▼
        agent loop continues
        (max 2 retries)
```

### Supported toolchains (`verify/runner.py`)

Auto-detected from project config files:

| Language | Detection | Lint | Types | Tests |
|---|---|---|---|---|
| Python | `pyproject.toml` or `setup.py` | ruff | mypy | pytest |
| TypeScript | `tsconfig.json` | eslint | tsc --noEmit | — |
| JavaScript | `package.json` | eslint | — | — |
| Rust | `Cargo.toml` | cargo clippy | cargo clippy | cargo test |

Only files actually changed in the current turn are passed to the linter/type checker.
Test discovery maps `src/foo.py` → `tests/test_foo.py` automatically.

---

## 8. Context Management

Long sessions are handled with a two-tier strategy to prevent the model's context
window from overflowing.

### Auto-compact (soft limit — 60 messages)

When the conversation history exceeds 60 messages, agentX asks the LLM to write a
4–6 sentence summary of what happened, then replaces the old messages with that summary
while keeping the 20 most recent messages intact.

### Hard trim (safety net — 80 messages)

If auto-compact fails (e.g. model error), the history is hard-trimmed to the last 80
messages as a final safety net.

### Tool output truncation

Any tool output longer than 8,000 characters is truncated with a hint:

```
[...3241 chars truncated. Use read_file with start_line/end_line to read specific sections.]
```

---

## 9. Custom Instructions — agentX.md

Create an `agentX.md` file in your project root to give the agent project-specific
instructions. It is read at the start of every turn and injected into the system prompt.

```markdown
# agentX.md

## Stack
- Python 3.11, FastAPI, PostgreSQL, SQLAlchemy 2.x
- Tests: pytest with real DB (no mocks)

## Rules
- Never use print() for logging — use structlog
- All endpoints must have type annotations
- Run `make lint` before finishing any task
```

agentX.md is truncated at 8,000 characters if it gets too large.

---

## 10. Sub-Agents

For complex tasks, agentX can delegate a focused sub-task to a completely fresh agent
instance using the `run_subtask` tool.

```
Parent agent
  │
  │  "write unit tests for src/auth/middleware.py"
  ▼
run_subtask(description)
  │
  ├── spawns fresh ConversationHistory (temp SQLite DB)
  ├── calls asyncio.run(run_turn(...))   ← safe in asyncio.to_thread()
  ├── sub-agent has all 21 tools, clean context
  ├── one level deep only (_AGENT_IN_SUBTASK guard)
  └── returns summary + files_changed to parent
```

Sub-agents inherit `AGENT_PROJECT_ROOT` and `AGENT_TRUST_MODE` from the parent.
Their history is isolated (temporary DB, deleted on completion).
Sub-agents cannot spawn their own sub-agents (nesting blocked).

**Good uses for run_subtask:**
- Write unit tests for a specific module
- Refactor a directory without changing the public API
- Find and replace a symbol across the whole codebase
- Any focused, self-contained chunk of work

---

## 11. Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | `python --version` to check |
| [Ollama](https://ollama.ai) | latest | Runs the LLM locally |
| ripgrep (`rg`) | any | Used by `search_code` |

---

### Windows

```powershell
# 1. Install Ollama — download the installer from https://ollama.ai/download
#    or use winget:
winget install Ollama.Ollama

# 2. Install ripgrep
winget install BurntSushi.ripgrep.MSVC
# Then CLOSE and REOPEN PowerShell to refresh PATH

# 3. Pull the recommended model
ollama pull qwen2.5:14b

# 4. (Optional) Pull the embedding model for episodic memory
ollama pull nomic-embed-text
```

### macOS

```bash
brew install ollama ripgrep
brew services start ollama
ollama pull qwen2.5:14b
ollama pull nomic-embed-text     # optional — for episodic memory
```

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
sudo apt install ripgrep          # Debian/Ubuntu
sudo pacman -S ripgrep            # Arch

ollama pull qwen2.5:14b
ollama pull nomic-embed-text      # optional
```

---

### Install agentX

```bash
# Clone the repo and install locally (recommended for development)
git clone https://github.com/YOUR_USERNAME/agentX.git
cd agentX
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

Or install directly from GitHub:

```bash
# Core (chat, file, git, shell tools)
pip install "git+https://github.com/YOUR_USERNAME/agentX.git"

# With code indexing (vector search, symbol index)
pip install "git+https://github.com/YOUR_USERNAME/agentX.git#egg=local-coding-agent[index]"

# With web search
pip install "git+https://github.com/YOUR_USERNAME/agentX.git#egg=local-coding-agent[search]"

# Everything
pip install "git+https://github.com/YOUR_USERNAME/agentX.git#egg=local-coding-agent[all]"
```

### First run

```bash
cd /your/project
agentX
```

On the **very first run** in a project, agentX runs a two-question setup wizard and
saves the answers to `.agent/config.toml`:

```
╭─────────────── agentX — setup ─────────────────╮
│ agentX needs to know where your Ollama          │
│ instance is running.                            │
│ Press Enter to accept the default value.        │
│                                                 │
│   Local Ollama : http://localhost:11434         │
│   Remote server: http://192.168.x.x:11434       │
╰─────────────────────────────────────────────────╯

Ollama URL [http://localhost:11434]:    ← Enter for local, or type your server IP
Model name [qwen2.5:14b]:              ← Enter for default, or pick another model
```

After setup, all subsequent runs use the saved config automatically.

---

## 12. Configuration

### .agent/config.toml

This is the primary config file, created by setup and managed with `agent config`:

```toml
trust_mode  = "trusted"
ollama_url  = "http://localhost:11434"
chat_model  = "qwen2.5:14b"

# Commands allowed to bypass trust restrictions (glob patterns)
shell_allowlist = [
    "npm run *",
    "pytest *",
    "python -m pytest *",
]
```

**Changing settings:**

```bash
# Switch to a remote Ollama server
agent config ollama_url http://192.168.1.50:11434

# Switch model
agent config chat_model qwen2.5:14b

# Change trust mode
agent config trust_mode yolo

# View all current settings
agent config
```

**Or re-run the full setup wizard any time:**

```bash
agent init
```

### Env var overrides (one session only)

Env vars take precedence over config.toml and apply only to that process:

```bash
# Use a different server just for this session
OLLAMA_URL=http://server:11434 agentX

# Use a lighter model just for this run
CHAT_MODEL=llama3.1:8b agentX "quick refactor"
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | from config or `http://localhost:11434` | Ollama server URL |
| `CHAT_MODEL` | from config or `qwen2.5:14b` | Model for chat/reasoning |
| `EMBED_MODEL` | `nomic-embed-text` | Model for embeddings (episodic memory) |
| `OLLAMA_TIMEOUT` | `120` | LLM request timeout in seconds |
| `AGENT_TRUST_MODE` | from config or `trusted` | Override trust mode |

### Choosing a model

agentX requires a model with **reliable tool-call support**. Not all Ollama models
support this. Here are tested recommendations:

| Model | Size | Tool Support | Notes |
|---|---|---|---|
| `qwen2.5:14b` | 14B | ✅ Excellent | **Recommended default** — best balance of speed and capability |
| `qwen2.5:7b` | 7B | ✅ Good | Faster, slightly weaker reasoning |
| `llama3.1:8b` | 8B | ✅ Good | Very reliable tool calling, fast |
| `llama3.1:70b` | 70B | ✅ Excellent | Best reasoning, needs high-end GPU |
| `qwen2.5-coder:14b` | 14B | ⚠️ Partial | Strong code knowledge, may output tool calls as text instead of function calls |
| `qwen3-coder:30b` | 30B | ✅ Excellent | Best coding model when available |
| `mistral:7b` | 7B | ✅ Good | Reliable, widely available |
| `deepseek-coder-v2:16b` | 16B | ✅ Good | Excellent for code tasks |

> **Windows users:** If the agent responds with raw JSON like `{"name": "find_files", ...}`
> instead of actually running the tool, your model does not have proper tool-call API support
> via Ollama. Switch to `qwen2.5:14b` or `llama3.1:8b`.

```bash
ollama pull qwen2.5:14b
agent config chat_model qwen2.5:14b
agentX
```

---

## 13. CLI Reference

### `agentX` command

```bash
agentX                          # start interactive session
agentX "fix the login bug"      # one-shot task, then exit
agentX sessions                 # list past sessions
agentX resume <session-id>      # resume a saved session
agentX --trust yolo             # start with yolo trust mode
agentX --workspace /other/path  # point at a different project
```

### `agent` command (full CLI)

```bash
agent init                      # set up .agent/ and run first index
agent chat                      # interactive session (same as agentX)
agent chat --trust readonly     # session with readonly trust
agent run "add a health check"  # one-shot task
agent sessions                  # list saved sessions
agent resume <id>               # resume a session
agent rollback                  # undo last task (git checkpoint)
agent log                       # recent task traces
agent log --task <task-id>      # detailed trace for one task
agent index                     # force full code reindex
agent config                    # show all config values
agent config trust_mode yolo    # set a config value
agent config chat_model qwen2.5:14b
```

### Running tests (development)

```bash
# Windows — use --basetemp to avoid system Temp permission issues
pytest tests/unit -v --tb=short --basetemp=.\tmp_pytest

# macOS / Linux
pytest tests/unit -v --tb=short

# Skip tests that need optional [index] extras
pytest tests/unit -v --tb=short --basetemp=.\tmp_pytest \
  --ignore=tests/unit/test_chunker.py \
  --ignore=tests/unit/test_context_assembler.py \
  --ignore=tests/unit/test_context_budgeter.py \
  --ignore=tests/unit/test_orchestrator_nodes.py \
  --ignore=tests/unit/test_symbols.py
```

---

## 14. Slash Commands

Available inside any interactive session:

| Command | Description |
|---|---|
| `/help` | Show all slash commands |
| `/clear` | Clear the current session's message history (keeps the session ID) |
| `/compact` | Manually trigger history summarisation now |
| `/tools` | List all 21 available tools with descriptions |
| `/memory` | Show detected project conventions |
| `/status` | Show session info (trust mode, message count, model, workspace) |

Type `exit`, `quit`, or press `Ctrl+C` to leave and save the session.

---

## 15. Project Layout

```
.
├── pyproject.toml          ← Package definition, dependencies, tool config
├── README.md               ← This file
├── agentX.md               ← Your project's custom instructions for agentX
│
├── src/agent/              ← All source code (described in §2)
│
└── tests/
    ├── unit/               ← Fast unit tests (no LLM, no network)
    │   ├── test_llm_chat.py
    │   ├── test_llm_prompts.py
    │   ├── test_llm_schemas.py
    │   ├── test_memory_consolidator.py
    │   ├── test_memory_episodic.py
    │   ├── test_memory_project.py
    │   ├── test_memory_working.py
    │   ├── test_observability.py
    │   ├── test_registry.py
    │   ├── test_safety_approval.py
    │   ├── test_safety_policy.py
    │   ├── test_tools_files.py
    │   ├── test_tools_git.py
    │   ├── test_tools_search.py
    │   ├── test_tools_shell.py
    │   ├── test_tools_tests.py
    │   └── test_verifier.py
    └── integration/        ← Integration tests (real filesystem, git)
        └── test_loop_integration.py
```

### Dependencies

```toml
# Core (always installed)
langchain-core, langchain-ollama, langgraph
pydantic, httpx, typer, rich
aiosqlite, tomli-w

# [index] extra — code vector search
lancedb, tree-sitter, tree-sitter-languages, tiktoken, watchdog

# [serve] extra — REST API server
fastapi, uvicorn

# [search] extra — web search tool
duckduckgo-search

# [dev] extra — development tools
pytest, pytest-asyncio, pytest-cov, ruff, mypy
```

---

## 16. Known Limitations

| Limitation | Details | Workaround |
|---|---|---|
| Windows shell tests | 3 unit tests use Unix commands (`ls`, `true`, `cat`) and fail on Windows | Expected — they test Linux shell behaviour; all core functionality works |
| Groq provider | `langchain_groq` not installed by default; 2 Groq tests are skipped | Install with `pip install langchain-groq` + set `GROQ_API_KEY` |
| `qwen2.5-coder` tool calling | This model may output tool calls as plain JSON text instead of structured function calls | Use `qwen2.5:14b` or `llama3.1:8b` instead |
| Windows Temp permissions | `pytest` may fail with `PermissionError` on `C:\Users\...\Temp\pytest-of-*` | Use `pytest --basetemp=.\tmp_pytest` |
| `[index]` extra optional | Code vector search (semantic, symbol lookup) requires `lancedb` and `tree-sitter` | Install with `pip install -e ".[index]"` |
| First response slow | A 14B model takes 1–3 minutes to load into RAM on first query | Subsequent queries are much faster |
