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

# Per-user state lives here (credentials, prompt history, first-run marker).
USER_DIR = Path.home() / ".autointerp"
USER_ENV_PATH = USER_DIR / "credentials"


def load_env_files(
    project_env: Path | None = None, user_env: Path | None = None
) -> None:
    """Load env files with standard CLI precedence.

    Process environment > project ``.env`` > user ``~/.autointerp/credentials``.
    ``override=False`` makes first-loaded win and never clobbers real env
    vars, so loading project first then user implements the precedence.
    """
    if project_env is not None:
        load_dotenv(project_env, override=False)
    else:
        load_dotenv(override=False)
    load_dotenv(user_env or USER_ENV_PATH, override=False)


class MCPServerConfig(BaseModel):
    """A minimal MCP server config compatible with FastMCP's shape."""

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    transport: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    model_name: str = DEFAULT_MODEL
    max_iterations: int = 500
    stream: bool = True
    auto_approve: bool = False
    local_mode: bool = True
    skills_dir: str = "skills"
    default_skills: list[str] = Field(default_factory=list)
    mcpServers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    system_prompt: str | None = None
    # Retry final answers that leak internal identifiers (InvestigationSpec,
    # finalize_spec, …) to the user. On for interactive Stage-0 chat; off for
    # the investigation pipeline, whose audience reads run artifacts anyway.
    plain_language_guard: bool = False
    # LLM sampling temperature for the agent driver. None means "don't send a
    # temperature" → the provider's own default applies (1.0 for OpenAI and
    # Anthropic chat APIs). Set via `/temperature` or AUTOINTERP_TEMPERATURE.
    # OpenAI reasoning models (gpt-5 family, o-series) ignore it — they only
    # accept their default — so it is dropped for those (see model_select).
    temperature: float | None = None


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


def env_default_model() -> str:
    """The model to use when no config file is present.

    Honors ``AUTOINTERP_MODEL`` (set by the setup flow's save step) the same
    way ``configs/agent.yaml`` does via ``${AUTOINTERP_MODEL:-...}``, so a
    saved model choice applies from any working directory.
    """
    return os.environ.get("AUTOINTERP_MODEL", DEFAULT_MODEL)


def env_default_temperature() -> float | None:
    """Optional ``AUTOINTERP_TEMPERATURE`` override; None if unset or unparsable
    (so a stray value never crashes startup — it just falls back to default)."""
    raw = os.environ.get("AUTOINTERP_TEMPERATURE")
    if not raw or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AgentConfig:
    """Load config from YAML/JSON plus environment defaults."""

    load_env_files()
    raw = _substitute_env(_load_structured_file(Path(config_path)))
    if "temperature" not in raw:
        env_temp = env_default_temperature()
        if env_temp is not None:
            raw["temperature"] = env_temp
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
