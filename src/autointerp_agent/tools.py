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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

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
        self.mcp_servers = mcp_servers or {}
        self.mcp_client: Any | None = None
        for tool in create_builtin_tools(skill_registry):
            self.register_tool(tool)

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

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

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if tool_needs_approval(tool_name, arguments, auto_approve=self.auto_approve):
            return (
                f"Approval required before running `{tool_name}` with arguments: {arguments}",
                False,
            )
        tool = self.tools.get(tool_name)
        if tool is not None and tool.handler is not None:
            return await tool.handler(arguments)
        if self.mcp_client is not None:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            content = "\n".join(str(item) for item in getattr(result, "content", []) or [])
            return content, not getattr(result, "is_error", False)
        return f"Unknown tool: {tool_name}", False


def create_builtin_tools(skill_registry: SkillRegistry) -> list[ToolSpec]:
    from .stage0_tools import create_stage0_tools

    return [
        _plan_tool(),
        _list_skills_tool(skill_registry),
        _read_skill_tool(skill_registry),
        _bash_tool(),
        _read_file_tool(),
        _write_file_tool(),
        _edit_file_tool(),
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


async def _run_bash(args: dict[str, Any]) -> tuple[str, bool]:
    command = str(args.get("command", ""))
    work_dir = str(args.get("work_dir", "."))
    timeout = min(int(args.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)
    if not command:
        return "No command provided.", False
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s.", False
    output = _truncate((proc.stdout or "") + (proc.stderr or ""))
    return output or "(no output)", proc.returncode == 0


async def _read_file(args: dict[str, Any]) -> tuple[str, bool]:
    path = str(args.get("path", ""))
    offset = max(int(args.get("offset") or 1), 1)
    limit = int(args.get("limit") or 400)
    if not path:
        return "No path provided.", False
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


async def _write_file(args: dict[str, Any]) -> tuple[str, bool]:
    path = str(args.get("path", ""))
    content = str(args.get("content", ""))
    if not path:
        return "No path provided.", False
    if Path(path).exists() and _resolved(path) not in _files_read:
        return f"You must read {path} before overwriting it.", False
    _atomic_write(Path(path), content)
    _files_read.add(_resolved(path))
    return f"Wrote {len(content)} bytes to {path}", True


async def _edit_file(args: dict[str, Any]) -> tuple[str, bool]:
    path = str(args.get("path", ""))
    old = str(args.get("old_str", ""))
    new = str(args.get("new_str", ""))
    replace_all = bool(args.get("replace_all", False))
    if not path or not old:
        return "path and old_str are required.", False
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


def _bash_tool() -> ToolSpec:
    return ToolSpec(
        name="bash",
        description=(
            "Run a local shell command. Destructive or installing commands require approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "work_dir": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
        handler=_run_bash,
    )


def _read_file_tool() -> ToolSpec:
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
        handler=_read_file,
    )


def _write_file_tool() -> ToolSpec:
    return ToolSpec(
        name="write_file",
        description=(
            "Write a local file. Existing files must be read first and approval is required."
        ),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        handler=_write_file,
    )


def _edit_file_tool() -> ToolSpec:
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
        handler=_edit_file,
    )
