"""CLI entry point — works like Claude Code in a terminal.

Commands:
    agent chat      — persistent interactive session
    agent run       — one-shot task
    agent sessions  — list recent sessions
    agent resume    — resume a previous session by ID
    agent init      — initialise workspace and run first index
    agent index     — force full reindex
    agent rollback  — undo last git checkpoint
    agent config    — get/set trust mode and settings
    agent log       — show task traces

Slash commands (inside chat):
    /help     — show available commands
    /clear    — clear current session history
    /compact  — summarise and compress conversation
    /tools    — list available tools
    /memory   — show project conventions and recent episodes
    /status   — show current session info
"""
from __future__ import annotations

import asyncio
import os
import sys
import tomllib
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent.llm.chat import DEFAULT_MODEL

app = typer.Typer(
    name="forge",
    help="Forge — Local AI coding agent. Fully private, runs on Ollama.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")

_TOOL_ICONS = {
    "write_file": "✎",
    "edit_file": "✎",
    "read_file": "📄",
    "list_dir": "📁",
    "run_shell": "⚡",
    "search_code": "🔍",
    "run_tests": "🧪",
    "fetch_url": "🌐",
    "run_subtask": "🤖",
    "git_status": "git",
    "git_diff": "git",
    "git_add": "git",
    "git_commit": "git",
    "git_log": "git",
    "git_checkpoint": "git",
    "git_rollback": "git",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _workspace() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _agent_dir(ws: Path | None = None) -> Path:
    return (ws or _workspace()) / ".agent"


def _config_path(ws: Path | None = None) -> Path:
    return _agent_dir(ws) / "config.toml"


def _load_config(ws: Path | None = None) -> dict[str, Any]:
    p = _config_path(ws)
    return tomllib.loads(p.read_text()) if p.exists() else {}


def _save_config(cfg: dict[str, Any], ws: Path | None = None) -> None:
    import tomli_w  # type: ignore[import]
    p = _config_path(ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(tomli_w.dumps(cfg).encode())


def _apply_llm_config(cfg: dict[str, Any]) -> None:
    """Push ollama_url and chat_model from config into env vars for llm/chat.py."""
    if "ollama_url" in cfg:
        os.environ["OLLAMA_URL"] = str(cfg["ollama_url"])
    if "chat_model" in cfg:
        os.environ["CHAT_MODEL"] = str(cfg["chat_model"])


def _first_run_setup(ws: Path) -> dict[str, Any]:
    """Interactive setup wizard shown the first time agentX runs in a project."""
    console.print(Panel(
        "[bold]First-time setup[/]\n\n"
        "agentX needs to know where your Ollama instance is running.\n"
        "Press [bold]Enter[/] to accept the default shown in brackets.\n\n"
        "  [dim]Local Ollama :[/]  http://localhost:11434\n"
        "  [dim]Remote server:[/]  http://192.168.x.x:11434",
        title="agentX — setup",
        border_style="cyan",
    ))

    url = typer.prompt(
        "\nOllama URL",
        default="http://localhost:11434",
    ).strip()

    model = typer.prompt(
        "Model name",
        default=DEFAULT_MODEL,
    ).strip()

    cfg: dict[str, Any] = {
        "trust_mode": "trusted",
        "ollama_url": url,
        "chat_model": model,
    }
    try:
        _save_config(cfg, ws)
    except ImportError:
        _config_path(ws).write_text(
            f'trust_mode = "trusted"\nollama_url = "{url}"\nchat_model = "{model}"\n'
        )

    console.print(f"\n[green]Config saved[/] → {_config_path(ws)}\n")
    console.print(
        "[dim]To change later:[/]\n"
        f"  agent config ollama_url <URL>\n"
        f"  agent config chat_model <model>\n"
    )
    return cfg


def _get_history(ws: Path) -> Any:
    from agent.core.history import ConversationHistory
    _agent_dir(ws).mkdir(parents=True, exist_ok=True)
    return ConversationHistory(_agent_dir(ws) / "state.db")


def _get_logger(ws: Path) -> Any:
    from agent.observability.logger import EventLogger
    return EventLogger(_agent_dir(ws) / "logs.db")


def _short_id(sid: str) -> str:
    return sid[:8]


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------

def _print_tool_start(name: str, args: dict[str, Any]) -> None:
    icon = _TOOL_ICONS.get(name, "🔧")
    hint = (
        args.get("command")
        or args.get("path")
        or args.get("query")
        or args.get("url")
        or args.get("description", "")[:60]
        or ""
    )
    if len(hint) > 72:
        hint = hint[:69] + "..."
    label = f"{icon} {name}"
    if hint:
        label += f"  [dim]{hint}[/]"
    console.print(f"  {label}")


def _print_tool_end(name: str, output: str, ok: bool) -> None:
    if not ok:
        preview = output[:200]
        console.print(f"    [red]✗ Error:[/] {preview}")
    else:
        lines = [ln for ln in output.splitlines() if ln.strip()]
        if lines and name in {"run_shell", "run_tests", "run_subtask"}:
            preview = lines[-1][:120]
            if preview:
                console.print(f"    [dim]↳ {preview}[/]")


# ---------------------------------------------------------------------------
# Streaming callbacks factory
# ---------------------------------------------------------------------------

def _make_callbacks() -> tuple[Any, Any, Any, Any]:
    """Return (on_content_token, on_content, on_tool_start, on_tool_end).

    on_content_token and on_content share state so content is printed exactly once.
    """
    did_stream: list[bool] = [False]

    def on_content_token(token: str) -> None:
        if not did_stream[0]:
            did_stream[0] = True
            sys.stdout.write("\n")  # blank line before first token
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_content(text: str) -> None:
        if did_stream[0]:
            # Tokens were already printed raw; just add spacing
            did_stream[0] = False
            sys.stdout.write("\n\n")
            sys.stdout.flush()
        elif text.strip():
            # Non-streaming path — render as markdown
            console.print()
            try:
                console.print(Markdown(text))
            except Exception:
                console.print(text)
            console.print()

    return on_content_token, on_content, _print_tool_start, _print_tool_end


# ---------------------------------------------------------------------------
# Core async run
# ---------------------------------------------------------------------------

async def _do_turn(
    user_message: str,
    ws: Path,
    session_id: str,
    trust: str,
    history: Any,
) -> Any:
    from agent.core.loop import run_turn

    on_content_token, on_content, on_tool_start, on_tool_end = _make_callbacks()

    return await run_turn(
        user_message,
        workspace=ws,
        session_id=session_id,
        history=history,
        trust=trust,  # type: ignore[arg-type]
        on_content=on_content,
        on_content_token=on_content_token,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

def _handle_slash(
    cmd: str,
    session_id: str,
    history: Any,
    ws: Path,
    trust: str,
) -> None:
    parts = cmd.strip().split(maxsplit=1)
    name = parts[0].lower()

    if name == "/help":
        console.print(Panel(
            "  [bold]/help[/]     — show this message\n"
            "  [bold]/clear[/]    — clear session history (keep session ID)\n"
            "  [bold]/compact[/]  — summarise and compress the conversation\n"
            "  [bold]/tools[/]    — list all available tools\n"
            "  [bold]/memory[/]   — show project conventions\n"
            "  [bold]/status[/]   — show current session info\n\n"
            "  [dim]Type [bold]exit[/] or press Ctrl+C to quit and save session.[/]",
            title="Slash commands",
            border_style="dim",
        ))

    elif name == "/clear":
        history.clear_session(session_id)
        console.print("[dim]Session history cleared.[/]")

    elif name == "/tools":
        from agent.tools.registry import all_tools
        tools = all_tools()
        table = Table(title="Available tools", border_style="dim", show_header=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", max_width=65)
        for t in tools:
            desc = (t.description or "").strip().splitlines()[0][:80]
            table.add_row(t.name, desc)
        console.print(table)

    elif name == "/memory":
        from agent.memory.project import ProjectStore
        agent_dir = ws / ".agent"
        try:
            facts = ProjectStore(agent_dir / "state.db").all()
        except Exception:
            facts = []
        if facts:
            table = Table(title="Project conventions", border_style="dim")
            table.add_column("Key", style="cyan")
            table.add_column("Value")
            for f in facts:
                table.add_row(f.key, f.value)
            console.print(table)
        else:
            console.print("[dim]No project conventions detected. Run 'agent init' first.[/]")

    elif name == "/status":
        msg_count = history.message_count(session_id)
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        chat_model_name = os.environ.get("CHAT_MODEL", DEFAULT_MODEL)
        console.print(Panel(
            f"Session  : [dim]{session_id}[/]\n"
            f"Messages : {msg_count}\n"
            f"Trust    : [cyan]{trust}[/]\n"
            f"Ollama   : [cyan]{ollama_url}[/]\n"
            f"Model    : [cyan]{chat_model_name}[/]\n"
            f"Workspace: [cyan]{ws}[/]\n\n"
            "[dim]Change model/URL: agent config ollama_url <url>  |  agent config chat_model <model>[/]",
            title="Session status",
            border_style="dim",
        ))

    elif name == "/compact":
        _compact_session(session_id, history, ws, trust)

    else:
        console.print(f"[yellow]Unknown command:[/] {name!r} — type [bold]/help[/] for the list.")


def _compact_session(session_id: str, history: Any, ws: Path, trust: str) -> None:
    """Summarise the conversation with the LLM and replace history with the summary."""
    from agent.llm.chat import chat_model
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = history.load(session_id)
    if not msgs:
        console.print("[dim]Nothing to compact.[/]")
        return

    # Build a condensed transcript
    lines = []
    for m in msgs:
        role = m.__class__.__name__.replace("Message", "")
        content = str(m.content)[:300].replace("\n", " ")
        lines.append(f"{role}: {content}")
    transcript = "\n".join(lines[:60])  # cap input

    prompt = (
        "Summarise this conversation in 4-6 sentences, preserving:\n"
        "- the original task or request\n"
        "- key decisions made\n"
        "- files created or modified\n"
        "- current state / what still needs to be done\n\n"
        f"Conversation:\n{transcript}"
    )

    try:
        os.environ["AGENT_PROJECT_ROOT"] = str(ws)
        os.environ["AGENT_TRUST_MODE"] = trust
        model = chat_model(temperature=0.2)
        result = asyncio.run(model.ainvoke([HumanMessage(content=prompt)]))
        summary = result.content if isinstance(result.content, str) else str(result.content)

        history.clear_session(session_id)
        history.append(session_id, AIMessage(content=f"[Compacted session summary]\n\n{summary}"))

        console.print("[green]Session compacted.[/]")
        console.print()
        console.print(Markdown(summary))
        console.print()
    except Exception as exc:
        console.print(f"[yellow]Compact failed:[/] {exc}")


# ---------------------------------------------------------------------------
# Turn result display
# ---------------------------------------------------------------------------

def _print_turn_result(result: Any) -> None:
    if result.files_changed:
        console.print("[dim]Files changed:[/]")
        for f in result.files_changed:
            console.print(f"  [green]✓[/] {f}")
    if result.tool_calls_made > 0 or result.iterations > 1:
        console.print(
            f"[dim]({result.tool_calls_made} tool calls · {result.iterations} iterations)[/]"
        )
    if result.files_changed or result.tool_calls_made > 0:
        console.print()


# ---------------------------------------------------------------------------
# agent chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[Optional[str], typer.Option("--trust")] = None,
    session: Annotated[Optional[str], typer.Option("--session", "-s")] = None,
) -> None:
    """Persistent interactive session — remembers everything across turns."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    cfg = _load_config(ws)
    trust_mode = trust or cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode
    _apply_llm_config(cfg)

    history = _get_history(ws)
    resumed = False

    try:
        if session:
            info = history.get_session(session)
            if not info:
                err_console.print(f"Session {session!r} not found.")
                raise typer.Exit(1)
            session_id = session
            resumed = True
        else:
            session_id = history.create_session(str(ws))

        _session_banner(ws, trust_mode, session_id, resumed)

        if resumed:
            prior = history.load(session_id)
            msg_count = len([m for m in prior if m.__class__.__name__ == "HumanMessage"])
            console.print(f"[dim]Loaded {msg_count} previous message(s).[/]\n")

        while True:
            try:
                user_input = typer.prompt("", prompt_suffix="[you] > ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session saved. Resume with:[/]")
                console.print(f"  agent chat --session {session_id}")
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit", "bye"}:
                console.print(f"\n[dim]Session saved. Resume with:[/]")
                console.print(f"  agent chat --session {session_id}")
                break
            if text.lower() == "sessions":
                _print_sessions(history)
                continue
            if text.startswith("/"):
                _handle_slash(text, session_id, history, ws, trust_mode)
                continue

            if history.message_count(session_id) == 0:
                history.update_title(session_id, text[:60])

            console.print(Rule(style="dim"))
            try:
                result = asyncio.run(_do_turn(text, ws, session_id, trust_mode, history))
                _print_turn_result(result)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]")
            except Exception as exc:
                err_console.print(f"Error: {exc}")
    finally:
        history.close()


# ---------------------------------------------------------------------------
# agent run (one-shot)
# ---------------------------------------------------------------------------

@app.command()
def run(
    request: Annotated[str, typer.Argument(help="Coding task to execute")],
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[Optional[str], typer.Option("--trust")] = None,
    session: Annotated[Optional[str], typer.Option("--session", "-s")] = None,
) -> None:
    """Execute a one-shot coding task."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    cfg = _load_config(ws)
    trust_mode = trust or cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode
    _apply_llm_config(cfg)

    history = _get_history(ws)
    try:
        if session:
            info = history.get_session(session)
            if not info:
                err_console.print(f"Session {session!r} not found.")
                raise typer.Exit(1)
            session_id = session
        else:
            session_id = history.create_session(str(ws), title=request[:60])

        console.print(Panel(f"[bold]{request}[/]", title="Task", border_style="dim"))
        console.print()

        try:
            result = asyncio.run(_do_turn(request, ws, session_id, trust_mode, history))
            _print_turn_result(result)
            console.print(f"[dim]Session ID: {session_id}[/]")
            console.print(f"[dim]Continue: agent chat --session {session_id}[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            console.print(f"[dim]Resume: agent chat --session {session_id}[/]")
            raise typer.Exit(130)
        except Exception as exc:
            err_console.print(f"Task failed: {exc}")
            raise typer.Exit(1)
    finally:
        history.close()


# ---------------------------------------------------------------------------
# agent sessions
# ---------------------------------------------------------------------------

@app.command()
def sessions(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """List recent conversation sessions."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    history = _get_history(ws)
    try:
        _print_sessions(history, limit)
    finally:
        history.close()


def _print_sessions(history: Any, limit: int = 20) -> None:
    rows = history.list_sessions(limit)
    if not rows:
        console.print("[dim]No sessions yet.[/]")
        return
    table = Table(title="Recent sessions", show_header=True, border_style="dim")
    table.add_column("ID", style="dim", no_wrap=True, width=10)
    table.add_column("Title", max_width=55)
    table.add_column("Updated", width=20)
    for r in rows:
        sid = r.get("id", "?")
        title = r.get("title") or "[dim](untitled)[/]"
        updated = (r.get("updated_at") or "")[:16].replace("T", " ")
        table.add_row(_short_id(sid), title, updated)
    console.print(table)
    console.print("[dim]Resume: agent chat --session <ID>[/]")


# ---------------------------------------------------------------------------
# agent resume
# ---------------------------------------------------------------------------

@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="Session ID to resume (8+ chars)")],
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[Optional[str], typer.Option("--trust")] = None,
) -> None:
    """Resume a previous session by its ID."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    history = _get_history(ws)
    try:
        all_sess = history.list_sessions(100)
        matches = [s for s in all_sess if s["id"].startswith(session_id)]
        if not matches:
            err_console.print(f"No session starting with {session_id!r}")
            raise typer.Exit(1)
        full_id = matches[0]["id"]
    finally:
        history.close()

    _forge_chat(ws, session_id=full_id)


# ---------------------------------------------------------------------------
# agent init
# ---------------------------------------------------------------------------

@app.command()
def init(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[str, typer.Option("--trust")] = "trusted",
) -> None:
    """Initialise .agent/ and run first index."""
    ws = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    agent_dir = _agent_dir(ws)
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Load existing config so we don't overwrite existing ollama_url/chat_model
    existing = _load_config(ws)

    console.print(Panel(
        "Configure the Ollama connection for this project.\n"
        "Press [bold]Enter[/] to keep the current value shown in brackets.",
        title="agent init — LLM setup",
        border_style="cyan",
    ))

    url = typer.prompt(
        "Ollama URL",
        default=existing.get("ollama_url", "http://localhost:11434"),
    ).strip()

    model = typer.prompt(
        "Model name",
        default=existing.get("chat_model", DEFAULT_MODEL),
    ).strip()

    cfg: dict[str, Any] = {
        "trust_mode": trust,
        "ollama_url": url,
        "chat_model": model,
    }
    try:
        _save_config(cfg, ws)
    except ImportError:
        _config_path(ws).write_text(
            f'trust_mode = "{trust}"\nollama_url = "{url}"\nchat_model = "{model}"\n'
        )

    _apply_llm_config(cfg)
    console.print(Panel(
        f"[green]Initialised[/] .agent/ in {ws}\n\n"
        f"  Ollama URL : [cyan]{url}[/]\n"
        f"  Model      : [cyan]{model}[/]\n"
        f"  Trust mode : [cyan]{trust}[/]\n\n"
        "Change any value with: [bold]agent config <key> <value>[/]",
        title="agent init",
    ))

    try:
        from agent.memory.project import ProjectStore
        store = ProjectStore(agent_dir / "state.db")
        facts = store.detect_and_persist(ws)
        if facts:
            console.print(f"Detected {len(facts)} project convention(s):")
            for f in facts:
                console.print(f"  [cyan]{f.key}[/]: {f.value}")
    except Exception as exc:
        console.print(f"[yellow]Convention detection skipped:[/] {exc}")

    console.print("Indexing project files…")
    try:
        os.environ["AGENT_PROJECT_ROOT"] = str(ws)
        asyncio.run(_do_index(ws, agent_dir))
        console.print("[green]Index complete.[/]")
    except Exception as exc:
        console.print(f"[yellow]Index skipped:[/] {exc}")

    console.print("\nRun [bold]agent chat[/] to start a session.")


async def _do_index(workspace: Path, agent_dir: Path) -> None:
    from agent.code_index import index_project
    await index_project(workspace, agent_dir)


# ---------------------------------------------------------------------------
# agent index
# ---------------------------------------------------------------------------

@app.command()
def index(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Force a full reindex of the project."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    agent_dir = _agent_dir(ws)
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    console.print("Reindexing…")
    try:
        asyncio.run(_do_index(ws, agent_dir))
        console.print("[green]Done.[/]")
    except Exception as exc:
        err_console.print(f"Index failed: {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# agent rollback
# ---------------------------------------------------------------------------

@app.command()
def rollback(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Undo last task via git checkpoint."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    from agent.tools.git import git_rollback as _git_rollback
    result = _git_rollback.invoke({})
    if "error" in str(result).lower() or "fail" in str(result).lower():
        err_console.print(str(result))
        raise typer.Exit(1)
    console.print(f"[green]Rolled back:[/] {result}")


# ---------------------------------------------------------------------------
# agent log
# ---------------------------------------------------------------------------

@app.command(name="log")
def log_cmd(
    task: Annotated[Optional[str], typer.Option("--task", "-t")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Show recent task traces."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    logger = _get_logger(ws)
    try:
        if task:
            task_data = logger.get_task(task)
            if task_data is None:
                err_console.print(f"Task {task!r} not found.")
                raise typer.Exit(1)
            events = logger.get_events(task)
            from agent.observability.tracer import format_trace
            console.print(format_trace(task_data, events))
        else:
            tasks = logger.list_tasks(limit=limit)
            if not tasks:
                console.print("No tasks logged yet.")
                return
            table = Table(title="Recent tasks", show_header=True)
            table.add_column("Task ID", style="dim", no_wrap=True)
            table.add_column("Request", max_width=50)
            table.add_column("Outcome")
            for t in tasks:
                colour = {"success": "green", "failure": "red", "partial": "yellow"}.get(
                    t.get("outcome", ""), "white"
                )
                table.add_row(
                    (t.get("task_id") or "?")[:12],
                    (t.get("request") or "")[:50],
                    f"[{colour}]{t.get('outcome', '?')}[/{colour}]",
                )
            console.print(table)
    finally:
        logger.close()


# ---------------------------------------------------------------------------
# agent config
# ---------------------------------------------------------------------------

@app.command(name="config")
def config_cmd(
    key: Annotated[Optional[str], typer.Argument()] = None,
    value: Annotated[Optional[str], typer.Argument()] = None,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Get or set agent configuration values."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    cfg = _load_config(ws)

    if key is None:
        if not cfg:
            console.print("No configuration found. Run [bold]agent init[/] first.")
            return
        table = Table(title=f"Config ({_config_path(ws)})")
        table.add_column("Key")
        table.add_column("Value")
        for k, v in cfg.items():
            table.add_row(k, str(v))
        console.print(table)
        return

    if value is None:
        if key not in cfg:
            err_console.print(f"Key {key!r} not found.")
            raise typer.Exit(1)
        console.print(cfg[key])
        return

    cfg[key] = value
    try:
        _save_config(cfg, ws)
    except ImportError:
        lines = "\n".join(f'{k} = "{v}"' for k, v in cfg.items())
        _config_path(ws).write_text(lines + "\n")
    console.print(f"[green]Set[/] {key} = {value}")


# ---------------------------------------------------------------------------
# agent serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[str, typer.Option("--trust")] = "trusted",
) -> None:
    """Start the agent as an HTTP API server."""
    try:
        import uvicorn  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        err_console.print(
            "fastapi/uvicorn not installed. Run: pip install 'forge[serve]'"
        )
        raise typer.Exit(1)

    ws = Path(workspace).resolve() if workspace else _workspace()
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust

    console.print(Panel(
        f"[bold]Agent API server[/]\n"
        f"Workspace : [cyan]{ws}[/]\n"
        f"Trust     : [cyan]{trust}[/]\n"
        f"Listening : [cyan]http://{host}:{port}[/]",
        title="forge serve",
        border_style="dim",
    ))

    import uvicorn  # type: ignore[import-untyped]
    uvicorn.run("agent.server:app", host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Session banner
# ---------------------------------------------------------------------------

def _session_banner(ws: Path, trust: str, session_id: str, resumed: bool = False) -> None:
    action = "Resumed" if resumed else "New session"
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    chat_model_name = os.environ.get("CHAT_MODEL", DEFAULT_MODEL)
    console.print(Panel(
        f"[bold]Local coding agent[/]   [dim]({action})[/]\n"
        f"Workspace : [cyan]{ws}[/]\n"
        f"Ollama    : [cyan]{ollama_url}[/]\n"
        f"Model     : [cyan]{chat_model_name}[/]\n"
        f"Trust     : [cyan]{trust}[/]\n"
        f"Session   : [dim]{session_id}[/]\n\n"
        "Type your task.  [dim][bold]/help[/] for commands · [bold]exit[/] to quit[/]",
        title="⚒  Forge",
        border_style="cyan",
    ))


# ---------------------------------------------------------------------------
# agentX entry point
# ---------------------------------------------------------------------------

def forge_main() -> None:
    """Entry point for the `forge` command.

    Usage:
        forge                    → interactive chat
        forge "fix the bug"      → one-shot task
        forge sessions           → list saved sessions
        forge resume <id>        → resume a session
    """
    args = sys.argv[1:]
    ws = _workspace()
    _agent_dir(ws).mkdir(parents=True, exist_ok=True)

    # First-run setup: if no config exists, ask for Ollama URL and model
    cfg = _load_config(ws)
    if not cfg:
        cfg = _first_run_setup(ws)
    _apply_llm_config(cfg)

    if not args:
        _forge_chat(ws)
    elif args[0] == "sessions":
        history = _get_history(ws)
        try:
            _print_sessions(history)
        finally:
            history.close()
    elif args[0] == "resume" and len(args) >= 2:
        _forge_chat(ws, session_id=args[1])
    elif args[0] in {"help", "--help", "-h"}:
        console.print(Panel(
            "[bold]forge[/] — Local AI coding agent\n\n"
            "  [bold]forge[/]                    start interactive session\n"
            "  [bold]forge[/] [italic]\"do something\"[/]    one-shot task\n"
            "  [bold]forge sessions[/]            list saved sessions\n"
            "  [bold]forge resume[/] [italic]<id>[/]        resume a session\n\n"
            "Inside a session, slash commands are available — type [bold]/help[/].\n\n"
            "LLM config (saved in .agent/config.toml):\n"
            "  forge config ollama_url http://localhost:11434\n"
            f"  forge config chat_model {DEFAULT_MODEL}\n\n"
            "Or override for one session with env vars:\n"
            "  OLLAMA_URL=http://server:11434 forge\n"
            f"  CHAT_MODEL=llama3.1:8b forge",
            title="forge help",
            border_style="dim",
        ))
    else:
        task = " ".join(args)
        _forge_run(ws, task)


def _forge_chat(ws: Path, session_id: str | None = None) -> None:
    cfg = _load_config(ws)
    trust_mode = cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode
    _apply_llm_config(cfg)

    history = _get_history(ws)
    resumed = False

    try:
        if session_id:
            all_sess = history.list_sessions(100)
            matches = [s for s in all_sess if s["id"].startswith(session_id)]
            if not matches:
                err_console.print(f"No session starting with {session_id!r}")
                return
            session_id = matches[0]["id"]
            resumed = True
        else:
            session_id = history.create_session(str(ws))

        _session_banner(ws, trust_mode, session_id, resumed)

        if resumed:
            prior = history.load(session_id)
            msg_count = len([m for m in prior if m.__class__.__name__ == "HumanMessage"])
            console.print(f"[dim]Loaded {msg_count} previous message(s).[/]\n")

        while True:
            try:
                user_input = input("[you] > ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print(f"\n[dim]Resume with: forge resume {session_id[:8]}[/]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                console.print(f"[dim]Resume with: forge resume {session_id[:8]}[/]")
                break
            if user_input.lower() == "sessions":
                _print_sessions(history)
                continue
            if user_input.startswith("/"):
                _handle_slash(user_input, session_id, history, ws, trust_mode)
                continue

            if history.message_count(session_id) == 0:
                history.update_title(session_id, user_input[:60])

            console.print(Rule(style="dim"))
            try:
                result = asyncio.run(_do_turn(user_input, ws, session_id, trust_mode, history))
                _print_turn_result(result)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]")
            except Exception as exc:
                err_console.print(f"Error: {exc}")
    finally:
        history.close()


def _forge_run(ws: Path, task: str) -> None:
    cfg = _load_config(ws)
    trust_mode = cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode
    _apply_llm_config(cfg)

    history = _get_history(ws)
    try:
        session_id = history.create_session(str(ws), title=task[:60])
        console.print(Panel(f"[bold]{task}[/]", title="⚒  Forge", border_style="cyan"))
        console.print()
        try:
            result = asyncio.run(_do_turn(task, ws, session_id, trust_mode, history))
            _print_turn_result(result)
            console.print(f"[dim]Continue: forge resume {session_id[:8]}[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
        except Exception as exc:
            err_console.print(f"Task failed: {exc}")
    finally:
        history.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
