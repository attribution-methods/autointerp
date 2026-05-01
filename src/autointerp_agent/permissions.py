"""Approval policy for local tools."""

from __future__ import annotations

import shlex


DESTRUCTIVE_COMMANDS = {
    "rm",
    "rmdir",
    "mv",
    "git reset",
    "git checkout",
    "git clean",
    "chmod",
    "chown",
    "dd",
}


def command_needs_approval(command: str) -> bool:
    stripped = command.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in DESTRUCTIVE_COMMANDS):
        return True
    if ">" in stripped or ">>" in stripped:
        return True
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return True
    installer = parts and parts[0] in {"sudo", "pip", "uv", "npm", "cargo", "python"}
    return bool(installer and "install" in parts)


def tool_needs_approval(tool_name: str, arguments: dict, auto_approve: bool = False) -> bool:
    if auto_approve:
        return False
    if tool_name == "bash":
        return command_needs_approval(str(arguments.get("command", "")))
    if tool_name in {"write_file", "edit_file"}:
        return True
    return False
