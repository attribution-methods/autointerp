"""Configuration loading for the autointerp agent.

This module follows the ML Intern pattern of a small config object with optional
MCP server definitions, but keeps defaults domain-specific and local-first.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
DEFAULT_CONFIG_PATH = Path("configs/agent.yaml")


class MCPServerConfig(BaseModel):
    """A minimal MCP server config compatible with FastMCP's shape."""

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    transport: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    model_name: str = DEFAULT_MODEL
    max_iterations: int = 60
    stream: bool = True
    auto_approve: bool = False
    local_mode: bool = True
    skills_dir: str = "skills"
    default_skills: list[str] = Field(default_factory=list)
    mcpServers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    system_prompt: str | None = None


def _load_structured_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return data


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        pattern = r"\$\{([^}:]+)(?::(-)?([^}]*))?\}"

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            has_default = match.group(2) is not None
            default = match.group(3) if has_default else None
            env_value = os.environ.get(name)
            if env_value is not None:
                return env_value
            if has_default:
                return default or ""
            raise ValueError(f"Environment variable {name!r} is not set")

        return re.sub(pattern, repl, value)
    if isinstance(value, dict):
        return {key: _substitute_env(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    return value


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AgentConfig:
    """Load config from YAML/JSON plus environment defaults."""

    load_dotenv(override=False)
    raw = _substitute_env(_load_structured_file(Path(config_path)))
    if "model" in raw and "model_name" not in raw:
        raw["model_name"] = raw.pop("model")
    if "name" in raw:
        raw.pop("name", None)
    if "version" in raw:
        raw.pop("version", None)
    raw.pop("description", None)
    raw.pop("tool_modules", None)
    raw.pop("default_loop", None)
    return AgentConfig.model_validate(raw)
