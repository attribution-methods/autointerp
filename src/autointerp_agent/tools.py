"""Tool router and local tools.

The structure is adapted from Hugging Face ML Intern's ToolRouter and local
tools, with the built-in tool set specialized for autointerp work.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from autointerp.utils.paths import is_within, normalize_safe_path

from .permissions import tool_needs_approval
from .skills import SkillRegistry

ToolHandler = Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]

MAX_OUTPUT_CHARS = 25_000
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 36_000
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")
_files_read: set[str] = set()


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler | None = None
    # Backwards-compat aliases. ``register_tool`` indexes them so the tool
    # can be found under any of these names. Lifted from claude-code Tool.ts —
    # rename a tool without breaking older specs / transcripts.
    aliases: list[str] = field(default_factory=list)

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRouter:
    def __init__(
        self,
        skill_registry: SkillRegistry,
        auto_approve: bool = False,
        mcp_servers: dict[str, Any] | None = None,
    ):
        self.skill_registry = skill_registry
        self.auto_approve = auto_approve
        self.tools: dict[str, ToolSpec] = {}
        self._aliases: dict[str, str] = {}
        self.mcp_servers = mcp_servers or {}
        self.mcp_client: Any | None = None
        # Range-restriction for write_file / edit_file. Investigation runs
        # set this to RunHandle.writable_roots(); REPL/Stage 0 leave it None
        # (write anywhere). Off by default — opt in per-context.
        self.writable_roots: list[Path] | None = None
        # Run directory for resolving relative paths in write_file.
        self.run_dir: Path | None = None
        # Optional scratch dir for the bash watchdog to spool stdout/stderr.
        self.scratch_dir: Path | None = None
        for tool in create_builtin_tools(skill_registry, router=self):
            self.register_tool(tool)

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool
        for alias in tool.aliases:
            self._aliases[alias] = tool.name

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self.tools.values()]

    async def __aenter__(self) -> "ToolRouter":
        if self.mcp_servers:
            await self._try_register_mcp_tools()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(exc_type, exc, tb)

    async def _try_register_mcp_tools(self) -> None:
        try:
            from fastmcp import Client
        except Exception:
            return
        try:
            self.mcp_client = Client({"mcpServers": self.mcp_servers})
            await self.mcp_client.__aenter__()
            await self.mcp_client.initialize()
            for tool in await self.mcp_client.list_tools():
                self.register_tool(
                    ToolSpec(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=tool.inputSchema,
                        handler=None,
                    )
                )
        except Exception:
            self.mcp_client = None

    def _resolve_tool_name(self, tool_name: str) -> str:
        if tool_name in self.tools:
            return tool_name
        return self._aliases.get(tool_name, tool_name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        canonical = self._resolve_tool_name(tool_name)
        if tool_needs_approval(canonical, arguments, auto_approve=self.auto_approve):
            return (
                f"Approval required before running `{canonical}` with arguments: {arguments}",
                False,
            )
        tool = self.tools.get(canonical)
        if tool is not None and tool.handler is not None:
            return await tool.handler(arguments)
        if self.mcp_client is not None:
            result = await self.mcp_client.call_tool(canonical, arguments)
            content = "\n".join(str(item) for item in getattr(result, "content", []) or [])
            return content, not getattr(result, "is_error", False)
        return f"Unknown tool: {canonical}", False


def create_builtin_tools(
    skill_registry: SkillRegistry, router: "ToolRouter | None" = None
) -> list[ToolSpec]:
    from .stage0_tools import create_stage0_tools

    return [
        _plan_tool(),
        _list_skills_tool(skill_registry),
        _read_skill_tool(skill_registry),
        _bash_tool(router),
        _read_file_tool(router),
        _write_file_tool(router),
        _edit_file_tool(router),
        *create_stage0_tools(),
    ]


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    text = _ANSI_RE.sub("", text)
    if len(text) <= limit:
        return text
    head = text[: limit // 4]
    tail = text[-(limit - len(head)) :]
    return f"{head}\n\n... omitted {len(text) - limit:,} chars ...\n\n{tail}"


def _resolved(path: str) -> str:
    return str(Path(path).resolve())


def _resolve_agent_path(path: str, router: "ToolRouter | None") -> str:
    """Resolve a possibly-relative agent path against run_dir or cwd.

    Handles both run-dir-relative (``scripts/foo.py``) and repo-root-relative
    (``active_runs/…/scripts/foo.py``) without doubling the prefix.
    """
    if Path(path).is_absolute() or router is None or router.run_dir is None:
        return path
    from_run = router.run_dir / path
    from_cwd = Path.cwd() / path
    if router.writable_roots is not None:
        in_run = is_within(str(from_run), router.writable_roots)
        in_cwd = is_within(str(from_cwd), router.writable_roots)
        if in_cwd and not in_run:
            return str(from_cwd)
        if in_run:
            return str(from_run)
    if from_cwd.exists() and not from_run.exists():
        return str(from_cwd)
    return str(from_run)


DEFAULT_STALL_SECONDS = 600  # kill on no-output stall (10 min). Stalled
# bash is the prior pain mode (nohup pipelines lying about completion).


def _bash_handler(router: "ToolRouter | None") -> ToolHandler:
    async def handler(args: dict[str, Any]) -> tuple[str, bool]:
        command = str(args.get("command", ""))
        work_dir = str(args.get("work_dir", "."))
        timeout = min(int(args.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)
        stall_seconds = int(args.get("stall_seconds") or DEFAULT_STALL_SECONDS)
        if not command:
            return "No command provided.", False
        scratch_dir = router.scratch_dir if router is not None else None
        return await asyncio.to_thread(
            _run_bash_watched, command, work_dir, timeout, stall_seconds, scratch_dir
        )

    return handler


def _run_bash_watched(
    command: str,
    work_dir: str,
    timeout: int,
    stall_seconds: int,
    scratch_dir: Path | None,
) -> tuple[str, bool]:
    """Run a bash command with both a hard timeout and a stall watchdog.

    The watchdog kills the process if it produces no output for
    ``stall_seconds``. This catches the prior pain mode (nohup pipelines
    backgrounded by mistake, hung subprocesses behind a pipe). The hard
    ``timeout`` remains the absolute upper bound.

    When ``scratch_dir`` is provided, full stdout/stderr is also streamed to
    a heartbeat log so a separate terminal can ``tail -f`` long jobs without
    waiting for them to finish.
    """
    start = time.monotonic()
    deadline = start + timeout
    last_output_at = start

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    heartbeat_path: Path | None = None
    if scratch_dir is not None:
        try:
            scratch_dir.mkdir(parents=True, exist_ok=True)
            heartbeat_path = scratch_dir / f"bash_{proc.pid}.log"
        except Exception:
            heartbeat_path = None

    chunks: list[bytes] = []
    stalled = False
    timed_out = False

    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    os.set_blocking(fd, False)

    def _drain_to_eof() -> None:
        while True:
            try:
                tail = os.read(fd, 65536)
            except BlockingIOError:
                return
            except OSError:
                return
            if not tail:
                return
            chunks.append(tail)
            if heartbeat_path is not None:
                try:
                    with heartbeat_path.open("ab") as fh:
                        fh.write(tail)
                except Exception:
                    pass

    while True:
        try:
            data = os.read(fd, 65536)
        except BlockingIOError:
            data = b""
        except OSError:
            data = b""
        if data:
            chunks.append(data)
            last_output_at = time.monotonic()
            if heartbeat_path is not None:
                try:
                    with heartbeat_path.open("ab") as fh:
                        fh.write(data)
                except Exception:
                    pass
            # Loop again immediately — drain in tight bursts.
            continue
        if proc.poll() is not None:
            _drain_to_eof()
            break
        now = time.monotonic()
        if now >= deadline:
            timed_out = True
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            _drain_to_eof()
            break
        if now - last_output_at >= stall_seconds:
            stalled = True
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            _drain_to_eof()
            break
        time.sleep(0.05)

    elapsed = time.monotonic() - start
    output = _truncate(b"".join(chunks).decode("utf-8", errors="replace"))
    rc = proc.returncode
    if timed_out:
        return f"Command timed out after {timeout}s.\n{output}", False
    if stalled:
        return (
            f"Command killed after {stall_seconds}s stall (no stdout/stderr).\n"
            f"Elapsed: {elapsed:.1f}s. Heartbeat log: {heartbeat_path or '(none)'}\n"
            f"{output}",
            False,
        )
    if rc != 0:
        # Surface non-zero exits prominently — the model frequently misses
        # 'silent' failures when stderr is interleaved with stdout.
        head = command if len(command) <= 80 else command[:77] + "..."
        return f"[exit {rc}] {head}\n{output or '(no output)'}", False
    return output or "(no output)", True


def _read_file_handler(router: "ToolRouter | None" = None) -> ToolHandler:
    async def handler(args: dict[str, Any]) -> tuple[str, bool]:
        path = str(args.get("path", ""))
        offset = max(int(args.get("offset") or 1), 1)
        limit = int(args.get("limit") or 400)
        if not path:
            return "No path provided.", False
        path = _resolve_agent_path(path, router)
        p = Path(path)
        if not p.exists() or p.is_dir():
            return f"File not found or not readable: {path}", False
        text = p.read_text(errors="replace")
        _files_read.add(_resolved(path))
        lines = text.splitlines()
        selected = lines[offset - 1 : offset - 1 + limit]
        numbered = "\n".join(
            f"{i:>6}\t{line[:4000]}" for i, line in enumerate(selected, start=offset)
        )
        return numbered, True
    return handler


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _check_writable(router: "ToolRouter | None", path: str) -> str | None:
    """Return an error message if ``path`` is unsafe under ``router.writable_roots``.

    None routers, or routers with no roots set, allow any path (Stage 0 / REPL
    behavior is unchanged). Investigation runs opt in by setting writable_roots.
    Callers must resolve relative paths via ``_resolve_agent_path`` first.
    """
    safe = normalize_safe_path(path)
    if safe is None:
        return f"Refused unsafe path: {path!r}"
    if router is None or router.writable_roots is None:
        return None
    if not is_within(safe, router.writable_roots):
        roots = ", ".join(str(r) for r in router.writable_roots)
        return (
            f"write_file is range-restricted in this run. "
            f"{safe} is outside the allowed roots ({roots}). "
            "Use commit_artifact for typed artifacts; scripts/ and scratch/ for free files."
        )
    return None


def _write_file_handler(router: "ToolRouter | None") -> ToolHandler:
    async def handler(args: dict[str, Any]) -> tuple[str, bool]:
        path = str(args.get("path", ""))
        content = str(args.get("content", ""))
        if not path:
            return "No path provided.", False
        path = _resolve_agent_path(path, router)
        err = _check_writable(router, path)
        if err is not None:
            return err, False
        if Path(path).exists() and _resolved(path) not in _files_read:
            return f"You must read {path} before overwriting it.", False
        _atomic_write(Path(path), content)
        _files_read.add(_resolved(path))
        return f"Wrote {len(content)} bytes to {path}", True

    return handler


def _edit_file_handler(router: "ToolRouter | None") -> ToolHandler:
    async def handler(args: dict[str, Any]) -> tuple[str, bool]:
        path = str(args.get("path", ""))
        old = str(args.get("old_str", ""))
        new = str(args.get("new_str", ""))
        replace_all = bool(args.get("replace_all", False))
        if not path or not old:
            return "path and old_str are required.", False
        path = _resolve_agent_path(path, router)
        err = _check_writable(router, path)
        if err is not None:
            return err, False
        if _resolved(path) not in _files_read:
            return f"You must read {path} before editing it.", False
        p = Path(path)
        text = p.read_text()
        count = text.count(old)
        if count == 0:
            return "old_str not found.", False
        if count > 1 and not replace_all:
            return (
                f"old_str appears {count} times. Set replace_all=true or use a narrower string.",
                False,
            )
        updated = text.replace(old, new, -1 if replace_all else 1)
        _atomic_write(p, updated)
        return f"Edited {path} ({count if replace_all else 1} replacement(s))", True

    return handler


_current_plan: list[dict[str, str]] = []


async def _update_plan(args: dict[str, Any]) -> tuple[str, bool]:
    global _current_plan
    todos = args.get("todos", [])
    if not isinstance(todos, list):
        return "todos must be a list.", False
    valid = {"pending", "in_progress", "completed"}
    for todo in todos:
        if not isinstance(todo, dict) or todo.get("status") not in valid:
            return "Each todo must include id, content, and valid status.", False
    _current_plan = todos
    return "\n".join(f"- [{item['status']}] {item['content']}" for item in todos), True


def _plan_tool() -> ToolSpec:
    return ToolSpec(
        name="plan",
        description="Track progress on multi-step tasks. Each call replaces the whole plan.",
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        handler=_update_plan,
    )


def _list_skills_tool(skill_registry: SkillRegistry) -> ToolSpec:
    async def handler(_args: dict[str, Any]) -> tuple[str, bool]:
        return "\n".join(skill_registry.list_lines()), True

    return ToolSpec(
        name="list_skills",
        description="List available autointerp method skills.",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def _read_skill_tool(skill_registry: SkillRegistry) -> ToolSpec:
    async def handler(args: dict[str, Any]) -> tuple[str, bool]:
        name = str(args.get("name", ""))
        if name not in skill_registry.skills:
            return f"Unknown skill: {name}", False
        skill = skill_registry.get(name)
        return f"# {skill.name}\n\n{skill.body}", True

    return ToolSpec(
        name="read_skill",
        description="Read a specific skill's instructions.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=handler,
    )


def _bash_tool(router: "ToolRouter | None" = None) -> ToolSpec:
    return ToolSpec(
        name="bash",
        description=(
            "Run a local shell command. The watchdog kills the process if it produces "
            "no output for `stall_seconds` (default 600s); use this for long-running "
            "jobs that should be aborted on stall. Destructive commands require approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "work_dir": {"type": "string"},
                "timeout": {"type": "integer", "description": "hard deadline in seconds"},
                "stall_seconds": {
                    "type": "integer",
                    "description": "kill if no stdout/stderr for this long (default 600)",
                },
            },
            "required": ["command"],
        },
        handler=_bash_handler(router),
    )


def _read_file_tool(router: "ToolRouter | None" = None) -> ToolSpec:
    return ToolSpec(
        name="read_file",
        description="Read a local file with line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
        handler=_read_file_handler(router),
    )


def _write_file_tool(router: "ToolRouter | None" = None) -> ToolSpec:
    return ToolSpec(
        name="write_file",
        description=(
            "Write a local file. Existing files must be read first. In an investigation "
            "run, paths are range-restricted to scripts/, scratch/, and INVESTIGATION_LOG.md."
        ),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        handler=_write_file_handler(router),
    )


def _edit_file_tool(router: "ToolRouter | None" = None) -> ToolSpec:
    return ToolSpec(
        name="edit_file",
        description="Edit a local file by exact string replacement. File must be read first.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_str", "new_str"],
        },
        handler=_edit_file_handler(router),
    )
