"""Prompt templates as plain string constants.

Use `.format(**kwargs)` for substitution. No PromptTemplate, no LCEL chains.

Sections are grouped by orchestrator node. Each constant is a complete message
body that becomes a SystemMessage or HumanMessage; the caller wraps it in the
appropriate LangChain message type.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt (injected once per conversation)
# ---------------------------------------------------------------------------

SYSTEM = """\
You are a local AI coding agent running entirely on this machine.

Workspace: {project_root}
Trust mode: {trust_mode}

Your job is to complete coding tasks by planning, executing tools, verifying
results, and reporting what changed. You have access to file operations, shell
commands, search, test runners, and git tools.

Rules:
- Work only within the project root. Never access paths outside it.
- To CREATE or OVERWRITE a file always use the write_file tool — never use run_shell to write files.
- To MODIFY part of an existing file use the edit_file tool.
- Use run_shell only for commands (pip install, pytest, git, starting servers, etc.).
- Read a file with read_file before editing it.
- Run tests after editing code to verify correctness.
- Commit only to non-protected branches.
- When uncertain, choose the safest action and explain why.

Project conventions detected:
{conventions}
"""

# ---------------------------------------------------------------------------
# Planning node
# ---------------------------------------------------------------------------

PLAN = """\
Task: {request}

Retrieved context (use this to inform your plan, but do not quote it verbatim):
{context}

Available tools: {tool_names}

Produce a JSON plan with:
- "steps": ordered list of {{"description": str, "expected_tools": [str, ...]}}
- "rationale": one sentence explaining the overall approach

Be specific about which files you expect to read or modify.
"""

# ---------------------------------------------------------------------------
# Acting node — appended to the message list before invoking with tools
# ---------------------------------------------------------------------------

ACT = """\
You MUST call at least one tool to make progress. Do NOT respond with plain text.
Use a tool call. If a tool fails, read the error and retry with fixed arguments.

Current step: {step_description}
Expected tools for this step: {expected_tools}
Iteration: {iteration}/{max_iterations}
"""

ACT_NO_TOOLS_WARNING = """\
Your previous response had no tool calls. You MUST call a tool right now.
Do not explain or describe — just call the tool immediately.

Suggested tool(s): {expected_tools}

Examples:
- To create a file → call write_file with path and content arguments
- To run a command → call run_shell with the command argument
- To read a file → call read_file with the path argument

Current step: {step_description}
Iteration: {iteration}/{max_iterations}
"""

# ---------------------------------------------------------------------------
# Verifier feedback — injected as a HumanMessage when verification fails
# ---------------------------------------------------------------------------

VERIFY_FAILURE = """\
Verification failed after your last edit. Fix the reported issues before
continuing. Do not re-run the same failing command without making a change.

Verifier output:
{verifier_output}

Files changed so far: {files_changed}
Failures remaining before task is aborted: {retries_left}
"""

# ---------------------------------------------------------------------------
# Commit node — summarise the completed task
# ---------------------------------------------------------------------------

COMMIT = """\
The task is complete. Write one or two sentences summarising what was accomplished.
Use only the information provided below — do NOT invent file names or claim actions \
that are not listed.

Request: {request}
Actions taken: {action_count} tool call(s)
Outcome: {outcome}
Files actually modified: {files_changed}
"""

# ---------------------------------------------------------------------------
# Retrieval context wrapper (injected into the message list by retrieve_node)
# ---------------------------------------------------------------------------

CONTEXT_BLOCK = """\
--- retrieved context (read-only reference material) ---
{chunks}
--- end of retrieved context ---
"""

EPISODE_BLOCK = """\
--- similar past tasks (for reference only) ---
{episodes}
--- end of past tasks ---
"""
