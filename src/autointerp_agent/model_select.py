"""Interactive model + API-key selection for the autointerp CLIs.

Claude-Code-style onboarding: when the configured model's provider has no
API key in the environment, the CLI walks the user through an inline
provider picker, a masked key prompt, a **live model list fetched from the
provider's /models endpoint**, a one-token validation ping, and an optional
save to ``.env`` (which ``load_config`` already auto-loads via
``load_dotenv``). The same flow backs the ``/model`` REPL command.

Model-catalog convention: providers ship new models constantly, so nothing
here hardcodes "the" model list. The provider endpoint is the source of
truth (it reflects exactly what the active key can call); litellm's bundled
registry is the offline fallback; a tiny built-in list is the last resort;
and a free-text "custom" entry always accepts any litellm model string.

Pure logic (provider detection, list filtering, env persistence, error
formatting) is kept separate from the prompt_toolkit UI so it stays
unit-testable without a TTY.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rich.console import Console

from .config import DEFAULT_MODEL, USER_ENV_PATH, AgentConfig

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    key: str
    display: str
    env_vars: tuple[str, ...]  # first entry is the one we prompt for
    key_url: str


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        key="anthropic",
        display="Anthropic",
        env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        key_url="https://console.anthropic.com/settings/keys",
    ),
    "openai": Provider(
        key="openai",
        display="OpenAI",
        env_vars=("OPENAI_API_KEY",),
        key_url="https://platform.openai.com/api-keys",
    ),
    "openrouter": Provider(
        key="openrouter",
        display="OpenRouter",
        env_vars=("OPENROUTER_API_KEY",),
        key_url="https://openrouter.ai/keys",
    ),
}

PROVIDER_ORDER = ("anthropic", "openai", "openrouter")

# --- Local (HuggingFace open-weights via a local OpenAI-compatible server) ---
# litellm routes `hosted_vllm/<id>` to HOSTED_VLLM_API_BASE — i.e. a local
# vLLM (or any OpenAI-compatible) server, with no API key and no collision
# with real OpenAI. The user runs the server; we just point litellm at it.
LOCAL_PREFIX = "hosted_vllm/"
DEFAULT_LOCAL_API_BASE = "http://localhost:8000/v1"
LOCAL_API_BASE_ENV = "HOSTED_VLLM_API_BASE"

# Curated open-weights chat models with reliable tool-calling, all fitting a
# <=180 GB GPU in bf16 (~2 GB / billion params + KV headroom). The user can
# also type any HuggingFace id. Sizes are approximate bf16 weight footprints.
LOCAL_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-7B-Instruct", "7B · ~16 GB · fast, strong tool use"),
    ("meta-llama/Llama-3.1-8B-Instruct", "8B · ~16 GB · widely used (gated repo)"),
    ("Qwen/Qwen3-8B", "8B · ~16 GB · newer Qwen"),
    ("Qwen/Qwen2.5-14B-Instruct", "14B · ~28 GB"),
    ("Qwen/Qwen2.5-32B-Instruct", "32B · ~64 GB · excellent agent"),
    ("Qwen/Qwen3-32B", "32B · ~64 GB · newer Qwen"),
    ("meta-llama/Llama-3.3-70B-Instruct", "70B · ~140 GB · frontier-class"),
    ("Qwen/Qwen2.5-72B-Instruct", "72B · ~145 GB · top open agent"),
]


def is_local_model(model_name: str) -> bool:
    return (model_name or "").startswith(LOCAL_PREFIX)


def local_api_base() -> str:
    """The configured local-server endpoint (for display)."""
    return os.environ.get(LOCAL_API_BASE_ENV) or DEFAULT_LOCAL_API_BASE

# Last resort ONLY — shown when both the live endpoint and litellm's bundled
# registry are unavailable. The live fetch is the source of truth; do not
# treat this list as current. Update opportunistically when touching this file.
FALLBACK_MODELS: list[tuple[str, str]] = [
    ("anthropic/claude-sonnet-4-5", "repo default — all prior runs/evals used this"),
    ("anthropic/claude-fable-5", "Anthropic frontier"),
    ("anthropic/claude-opus-4-8", ""),
    ("anthropic/claude-sonnet-4-6", ""),
    ("anthropic/claude-haiku-4-5", "fast + cheap"),
    ("openai/gpt-5.1", ""),
    ("openrouter/anthropic/claude-sonnet-4.5", "Claude via OpenRouter (eval setup)"),
]


def provider_for_model(model_name: str) -> Provider | None:
    """Map a litellm model string to its provider, mirroring litellm routing.

    ``openrouter/...`` wins over the nested provider segment (the key that
    gets validated is OpenRouter's). Bare ``claude*`` / ``gpt*`` ids route to
    Anthropic / OpenAI the way litellm resolves them. Unknown providers
    return None, which callers treat as "cannot check — don't gate".
    """
    name = (model_name or "").lower()
    if name.startswith("openrouter/"):
        return PROVIDERS["openrouter"]
    prefix = name.split("/", 1)[0]
    if prefix in PROVIDERS:
        return PROVIDERS[prefix]
    if name.startswith("claude"):
        return PROVIDERS["anthropic"]
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return PROVIDERS["openai"]
    return None


# OpenAI's reasoning models (gpt-5 family, o-series) reject any temperature
# other than their default (1); every other model honors a custom temperature.
_NO_CUSTOM_TEMPERATURE = re.compile(r"(?:^|/)(?:gpt-5|o1|o3|o4)", re.IGNORECASE)


def model_supports_temperature(model_name: str) -> bool:
    """Whether a custom sampling temperature can be sent to this model.

    False for OpenAI reasoning models (gpt-5 family, o-series): their API only
    accepts the default and 400s on any other value. Used both to drop the
    param before it reaches the provider and to warn the user when they set
    one on such a model.
    """
    return _NO_CUSTOM_TEMPERATURE.search(model_name or "") is None


def missing_key_env(model_name: str) -> str | None:
    """Return the env var to prompt for if the model's provider has no key set.

    None means ready to go: either a key is present or the provider is
    unknown to us (litellm will surface its own error in that case).
    """
    provider = provider_for_model(model_name)
    if provider is None:
        return None
    if any(os.environ.get(var) for var in provider.env_vars):
        return None
    return provider.env_vars[0]


def _active_env_key(provider: Provider) -> tuple[str, str] | None:
    """(env var, value) for the provider's currently-set key, if any."""
    for var in provider.env_vars:
        value = os.environ.get(var)
        if value:
            return var, value
    return None


def _provider_key(provider: Provider) -> str | None:
    active = _active_env_key(provider)
    return active[1] if active else None


# ---------------------------------------------------------------------------
# Live model catalogs (provider /models endpoints)
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT = 8.0

# Trailing release-date suffixes: OpenAI style (-2025-11-13) and Anthropic
# style (-20251101).
_DATED_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")

_OPENAI_EXCLUDE = (
    "embedding", "whisper", "tts", "dall-e", "audio", "realtime",
    "moderation", "babbage", "davinci", "instruct", "search", "transcribe",
    "image", "codex", "ft:",
)


def _is_openai_chat_id(model_id: str) -> bool:
    low = model_id.lower()
    if any(token in low for token in _OPENAI_EXCLUDE):
        return False
    return low.startswith(("gpt-", "chatgpt-")) or re.match(r"^o\d", low) is not None


def _dedupe_dated(ids: Sequence[str]) -> list[str]:
    """Drop dated snapshots whose undated alias is also in the list."""
    present = set(ids)
    out: list[str] = []
    for model_id in ids:
        if _DATED_SUFFIX.search(model_id) and _DATED_SUFFIX.sub("", model_id) in present:
            continue
        out.append(model_id)
    return out


def _fetch_anthropic_models(api_key: str) -> list[tuple[str, str]]:
    import httpx

    resp = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        params={"limit": 50},
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    data.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return [(f"anthropic/{m['id']}", str(m.get("display_name", ""))) for m in data if m.get("id")]


def _fetch_openai_models(api_key: str) -> list[tuple[str, str]]:
    import httpx

    resp = httpx.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = [m for m in resp.json().get("data", []) if _is_openai_chat_id(str(m.get("id", "")))]
    data.sort(key=lambda m: m.get("created", 0), reverse=True)
    kept = _dedupe_dated([m["id"] for m in data])[:15]
    return [(f"openai/{model_id}", "") for model_id in kept]


# Families worth showing from OpenRouter's very large catalog; "custom…"
# covers the rest. Tilde-prefixed ids are OpenRouter's rolling-latest
# redirect aliases — callable, so kept.
_OPENROUTER_FAMILIES = (
    "anthropic/", "openai/", "google/", "meta-llama/", "qwen/",
    "deepseek/", "mistralai/", "x-ai/",
)


def _fetch_openrouter_models(api_key: str | None) -> list[tuple[str, str]]:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(
        "https://openrouter.ai/api/v1/models", headers=headers, timeout=_HTTP_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    rows = [
        m for m in data
        if str(m.get("id", "")).lstrip("~").startswith(_OPENROUTER_FAMILIES)
        # The agent sends tools on every call; hide models that cannot take
        # them (image gens, safety classifiers, …). Custom ids bypass this
        # and are caught by the validation ping instead.
        and "tools" in (m.get("supported_parameters") or [])
    ]
    rows.sort(key=lambda m: m.get("created", 0), reverse=True)
    return [(f"openrouter/{m['id']}", str(m.get("name", "")))
            for m in rows[:20]]


def _fetch_live_models(provider: Provider) -> list[tuple[str, str]]:
    api_key = _provider_key(provider)
    if provider.key == "anthropic":
        if not api_key:
            return []
        return _fetch_anthropic_models(api_key)
    if provider.key == "openai":
        if not api_key:
            return []
        return _fetch_openai_models(api_key)
    if provider.key == "openrouter":
        return _fetch_openrouter_models(api_key)  # endpoint works unauthenticated
    return []


def _registry_models(provider: Provider) -> list[tuple[str, str]]:
    """Offline fallback: litellm's bundled model registry (pip-release fresh)."""
    try:
        import litellm

        ids = list(litellm.models_by_provider.get(provider.key, []))
    except Exception:  # noqa: BLE001 — any registry weirdness → next fallback
        return []
    prefix = provider.key + "/"
    ids = [i[len(prefix):] if i.startswith(prefix) else i for i in ids]
    if provider.key == "anthropic":
        ids = [i for i in ids if i.startswith("claude")]
    elif provider.key == "openai":
        ids = [i for i in ids if _is_openai_chat_id(i)]
    elif provider.key == "openrouter":
        ids = [i for i in ids if i.startswith(_OPENROUTER_FAMILIES)]
    # No release dates in the registry; reverse-lexicographic at least puts
    # higher version numbers first within a family.
    ids = _dedupe_dated(sorted(set(ids), reverse=True))[:15]
    return [(prefix + i, "") for i in ids]


async def fetch_models_with_fallback(provider: Provider) -> tuple[list[tuple[str, str]], str]:
    """Live endpoint → litellm registry → built-in list. Returns (models, source)."""
    try:
        live = await asyncio.to_thread(_fetch_live_models, provider)
        if live:
            return live, "live"
    except Exception:  # noqa: BLE001 — offline/401/timeouts all degrade the same way
        pass
    registry = _registry_models(provider)
    if registry:
        return registry, "litellm registry"
    fallback = [
        (model_id, desc)
        for model_id, desc in FALLBACK_MODELS
        if (p := provider_for_model(model_id)) is not None and p.key == provider.key
    ]
    return fallback, "built-in fallback"


# ---------------------------------------------------------------------------
# .env persistence
# ---------------------------------------------------------------------------


def persist_to_env_file(updates: dict[str, str], env_path: Path | None = None) -> Path:
    """Write/replace ``VAR=value`` lines in ``.env`` (created 0600 if new).

    Existing unrelated lines are preserved; existing assignments for the same
    var are replaced in place. Values are also exported to ``os.environ`` so
    the current process picks them up immediately.
    """
    path = env_path or Path(".env")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if path.exists():
        lines = path.read_text().splitlines()
    remaining = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        var = stripped.split("=", 1)[0].strip()
        if var in remaining:
            lines[i] = f"{var}={remaining.pop(var)}"
    for var, value in remaining.items():
        lines.append(f"{var}={value}")
    path.write_text("\n".join(lines) + "\n")
    try:
        # Key-bearing file: tighten perms on every write, not just creation
        # (the file may have been hand-created with the user's umask).
        os.chmod(path, 0o600)
    except OSError:
        pass
    for var, value in updates.items():
        os.environ[var] = value
    return path


def env_file_is_gitignored(repo_root: Path | None = None) -> bool:
    root = repo_root or Path(".")
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return False
    patterns = {line.strip() for line in gitignore.read_text().splitlines()}
    return ".env" in patterns or "*.env" in patterns


# ---------------------------------------------------------------------------
# Validation + error formatting
# ---------------------------------------------------------------------------


_PING_MAX_TOKENS = 16

# The agent loop sends a tools array on every call (agent_loop.py), so the
# validation ping must too — a model that accepts plain chat but rejects
# tools would otherwise pass setup and fail on the first real turn.
_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Connectivity check; never call this.",
        "parameters": {"type": "object", "properties": {}},
    },
}

# Reasoning models (gpt-5 family, o-series, …) can spend the entire ping
# budget on hidden reasoning tokens; the provider then errors with a
# "max_tokens / output limit reached" BadRequest. That error still proves
# the key authenticated and the model ran — which is all the ping tests.
_BENIGN_PING_ERROR = re.compile(
    r"max_tokens|max_completion_tokens|output limit|finish the message",
    re.IGNORECASE,
)


def _is_benign_ping_error(exc: Exception) -> bool:
    kind = type(exc).__name__.lower()
    if "badrequest" not in kind and "invalidrequest" not in kind:
        return False
    return _BENIGN_PING_ERROR.search(str(exc)) is not None


async def validate_model(model_name: str) -> tuple[bool, str]:
    """Minimal completion ping. Returns (ok, short_error)."""
    import litellm

    # Keep litellm's "Give Feedback / Get Help" + debug-hint banners out of
    # the setup flow; we render the error ourselves.
    litellm.suppress_debug_info = True
    try:
        await litellm.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=_PING_MAX_TOKENS,
            tools=[_PING_TOOL],
            tool_choice="auto",
        )
        return True, ""
    except Exception as exc:  # noqa: BLE001 — every provider error lands here
        if _is_benign_ping_error(exc):
            return True, ""
        return False, format_llm_error(exc, model_name=model_name, hint=False)


def format_llm_error(exc: Exception, *, model_name: str | None = None, hint: bool = True) -> str:
    """One readable line for a provider/litellm exception (no traceback)."""
    kind = type(exc).__name__
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else kind
    # litellm stacks redundant "litellm.AuthenticationError: AuthenticationError:"
    # prefixes onto provider messages — drop them all, we print `kind` ourselves.
    first_line = re.sub(
        r"^(?:(?:litellm\.)?\w+(?:Error|Exception)\s*:\s*){1,3}", "", first_line
    ) or first_line
    if len(first_line) > 240:
        first_line = first_line[:237] + "..."
    target = f" [{model_name}]" if model_name else ""
    msg = f"{kind}{target}: {first_line}"
    if hint and "auth" in kind.lower():
        msg += "  (use /model to reconfigure the model or API key)"
    return msg


# ---------------------------------------------------------------------------
# prompt_toolkit inline UI
# ---------------------------------------------------------------------------


async def _select_async(
    title: str,
    options: Sequence[tuple[str, str]],
    *,
    default_index: int = 0,
) -> int | None:
    """Inline arrow-key selector. Returns the chosen index, or None on cancel."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    state = {"index": max(0, min(default_index, len(options) - 1))}

    def fragments():
        parts: list[tuple[str, str]] = [("class:title", f"{title}\n")]
        for i, (label, desc) in enumerate(options):
            selected = i == state["index"]
            cursor = "❯ " if selected else "  "
            style = "class:selected" if selected else "class:option"
            parts.append((style, f"{cursor}{label}"))
            if desc:
                parts.append(("class:desc", f"  {desc}"))
            parts.append(("", "\n"))
        parts.append(("class:hint", "↑/↓ move · enter select · esc cancel"))
        return parts

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event) -> None:
        state["index"] = (state["index"] - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(event) -> None:
        state["index"] = (state["index"] + 1) % len(options)

    @kb.add("enter")
    def _accept(event) -> None:
        event.app.exit(result=state["index"])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("q")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    for digit in range(1, min(9, len(options)) + 1):

        @kb.add(str(digit))
        def _jump(event, _i=digit - 1) -> None:
            event.app.exit(result=_i)

    app: Application[int | None] = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(fragments))])),
        key_bindings=kb,
        style=Style.from_dict(
            {
                "title": "bold",
                "selected": "bold ansicyan",
                "option": "",
                "desc": "ansibrightblack",
                "hint": "ansibrightblack italic",
            }
        ),
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
    )
    return await app.run_async()


async def _prompt_text(message: str, *, password: bool = False) -> str | None:
    """Single-line input; returns None on ctrl-c / ctrl-d / empty."""
    from prompt_toolkit import PromptSession

    try:
        value = await PromptSession().prompt_async(message, is_password=password)
    except (EOFError, KeyboardInterrupt):
        return None
    value = (value or "").strip()
    return value or None


_CUSTOM_PROMPT = "model id (litellm format, e.g. anthropic/claude-sonnet-4-5): "


async def select_provider(current: Provider | None) -> Provider | str | None:
    """Pick a provider, ``'local'`` (HuggingFace on a local server), or
    ``'custom'`` (free-text model id). None = cancel."""
    options = [
        ("Anthropic", "— Claude models (repo default provider)"),
        ("OpenAI", ""),
        ("OpenRouter", "— many providers through one key"),
        ("Local (HuggingFace)", "— open-weights model on your own GPU (vLLM)"),
        ("Custom model id", "— type any litellm model string"),
    ]
    default_index = PROVIDER_ORDER.index(current.key) if current is not None else 0
    choice = await _select_async("Select a provider", options, default_index=default_index)
    if choice is None:
        return None
    if choice == len(options) - 1:
        return "custom"
    if choice == len(options) - 2:
        return "local"
    return PROVIDERS[PROVIDER_ORDER[choice]]


async def select_local_model(current: str | None) -> str | None:
    """Pick a curated open-weights HF model id, or type a custom one."""
    cur_id = current[len(LOCAL_PREFIX):] if is_local_model(current or "") else None
    options: list[tuple[str, str]] = []
    default_index = 0
    for i, (model_id, desc) in enumerate(LOCAL_MODELS):
        marker = "  (current)" if model_id == cur_id else ""
        options.append((model_id, f"— {desc}{marker}"))
        if model_id == cur_id:
            default_index = i
    options.append(("custom…", "— type any HuggingFace model id"))
    choice = await _select_async(
        "Select an open-weights model", options, default_index=default_index
    )
    if choice is None:
        return None
    if choice == len(options) - 1:
        return await _prompt_text(
            "HuggingFace model id (e.g. Qwen/Qwen2.5-7B-Instruct): "
        )
    return LOCAL_MODELS[choice][0]


def _annotate(
    models: Sequence[tuple[str, str]], current: str
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for model_id, desc in models:
        notes = []
        if model_id == current:
            notes.append("current")
        if model_id == DEFAULT_MODEL:
            notes.append("repo default")
        text = f"— {desc}" if desc else ""
        if notes:
            text = (text + f"  ({', '.join(notes)})").strip()
        out.append((model_id, text))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_NONINTERACTIVE_HELP = (
    "Set {env_var} in the environment, in a .env file at the repo root, or "
    "in ~/.autointerp/credentials; pass --model to use a different provider; "
    "or run `autointerp` in a terminal for interactive setup."
)


async def ensure_model_ready(
    config: AgentConfig,
    console: Console,
    *,
    interactive: bool,
) -> AgentConfig | None:
    """Gate a CLI entry point on having a usable model + key.

    Returns the (possibly updated) config when ready, or None when the
    session cannot proceed. Never raises for provider problems.
    """
    env_var = missing_key_env(config.model_name)
    if env_var is None:
        return config
    if not interactive or not sys.stdin.isatty():
        console.print(
            f"[red]No API key found for model "
            f"'[bold]{config.model_name}[/bold]' (needs {env_var}).[/red]"
        )
        console.print(_NONINTERACTIVE_HELP.format(env_var=env_var))
        return None
    console.print(
        f"[yellow]No API key found for "
        f"'[bold]{config.model_name}[/bold]' (needs {env_var}).[/yellow] "
        "Let's set one up."
    )
    updated = await run_setup_flow(config, console)
    if updated is None:
        console.print(
            f"[yellow]Setup cancelled.[/yellow] Set {env_var} (env or .env) "
            "and rerun, or pass --model for a different provider."
        )
    return updated


async def run_setup_flow(config: AgentConfig, console: Console) -> AgentConfig | None:
    """Provider → key entry (if needed) → live model list → validation → save.

    The user can back out at any point (returns None at startup; the /model
    command treats None as "keep the old config").
    """
    while True:
        picked = await select_provider(provider_for_model(config.model_name))
        if picked is None:
            return None

        entered_key: str | None = None
        env_var: str | None = None
        model: str | None = None
        extra_env: dict[str, str] = {}

        if picked == "custom":
            model = await _prompt_text(_CUSTOM_PROMPT)
            if model is None:
                continue
            env_var = missing_key_env(model)
            if env_var is not None:
                provider = provider_for_model(model)
                assert provider is not None  # missing_key_env implies known provider
                console.print(f"[dim]{provider.display} keys: {provider.key_url}[/dim]")
                entered_key = await _prompt_text(f"{env_var}: ", password=True)
                if entered_key is None:
                    continue
                os.environ[env_var] = entered_key
        elif picked == "local":
            console.print(
                "[dim]Local models run on your own GPU via an OpenAI-compatible "
                "server (e.g. vLLM). No API key needed — just a reachable "
                "endpoint with tool-calling enabled.[/dim]"
            )
            model_id = await select_local_model(config.model_name)
            if model_id is None:
                continue
            base = await _prompt_text(
                f"server base URL [default {DEFAULT_LOCAL_API_BASE}]: "
            ) or DEFAULT_LOCAL_API_BASE
            os.environ[LOCAL_API_BASE_ENV] = base
            extra_env[LOCAL_API_BASE_ENV] = base
            model = LOCAL_PREFIX + model_id
            console.print(
                f"[dim]If nothing is serving it yet:[/dim] vllm serve {model_id} "
                f"--enable-auto-tool-choice --tool-call-parser hermes"
            )
        else:
            provider = picked
            active = _active_env_key(provider)
            needs_key_entry = active is None
            if active is not None:
                var, value = active
                tail = f"…{value[-4:]}" if len(value) >= 8 else "(set)"
                key_choice = await _select_async(
                    f"{var} is already set ({tail})",
                    [
                        ("Use the existing key", ""),
                        ("Enter a new key", "— replaces it; you can save it after"),
                    ],
                )
                if key_choice is None:
                    continue
                if key_choice == 1:
                    needs_key_entry = True
                    env_var = var
            if needs_key_entry:
                env_var = env_var or provider.env_vars[0]
                console.print(f"[dim]{provider.display} keys: {provider.key_url}[/dim]")
                entered_key = await _prompt_text(f"{env_var}: ", password=True)
                if entered_key is None:
                    continue
                os.environ[env_var] = entered_key

            with console.status(f"[dim]fetching {provider.display} model list…[/dim]"):
                models, source = await fetch_models_with_fallback(provider)
            if source != "live":
                console.print(
                    f"[yellow]live model list unavailable (check key / network) "
                    f"— showing {source}.[/yellow]"
                )
            options = _annotate(models, config.model_name)
            options.append(("custom…", "— type any litellm model id"))
            default_index = next(
                (i for i, (mid, _) in enumerate(options) if mid == config.model_name), 0
            )
            choice = await _select_async(
                "Select a model", options, default_index=default_index
            )
            if choice is None:
                if entered_key is not None and env_var is not None:
                    os.environ.pop(env_var, None)  # don't keep an unvetted key
                continue
            if choice == len(options) - 1:
                model = await _prompt_text(_CUSTOM_PROMPT)
                if model is None:
                    continue
            else:
                model = options[choice][0]

        console.print(f"[dim]model:[/dim] [bold]{model}[/bold]")

        with console.status(f"[dim]validating {model}…[/dim]"):
            ok, err = await validate_model(model)
        if not ok:
            if entered_key is not None and env_var is not None:
                # Don't leave a bad key poisoning the session environment.
                os.environ.pop(env_var, None)
            console.print(f"[red]✗ {err}[/red]")
            retry = await _select_async(
                "Validation failed — what next?",
                [
                    ("Try again", "— re-enter the key / pick another model"),
                    ("Continue anyway", "— use this config without validating"),
                    ("Quit setup", ""),
                ],
            )
            if retry == 0:
                continue  # back to the provider picker
            if retry == 1:
                if entered_key is not None and env_var is not None:
                    os.environ[env_var] = entered_key
            else:
                return None
        else:
            console.print(f"[green]✓ {model} responded[/green]")

        updates: dict[str, str] = {}
        if entered_key is not None and env_var is not None:
            updates[env_var] = entered_key
        updates.update(extra_env)  # e.g. HOSTED_VLLM_API_BASE for local models
        if model != config.model_name or updates:
            updates["AUTOINTERP_MODEL"] = model
        if updates:
            save = await _select_async(
                f"Save {', '.join(updates)} for future sessions?",
                [
                    (
                        "Yes — for this user",
                        f"— {USER_ENV_PATH} (works in any directory)",
                    ),
                    ("Yes — for this project only", "— ./.env"),
                    ("No", "— this session only"),
                ],
            )
            if save == 0:
                path = persist_to_env_file(updates, USER_ENV_PATH)
                console.print(f"[dim]saved to {path}[/dim]")
            elif save == 1:
                path = persist_to_env_file(updates)
                console.print(f"[dim]saved to {path}[/dim]")
                if not env_file_is_gitignored():
                    console.print(
                        "[yellow]warning: .env is not listed in .gitignore — "
                        "do not commit it.[/yellow]"
                    )
        return config.model_copy(update={"model_name": model})


__all__ = [
    "FALLBACK_MODELS",
    "LOCAL_MODELS",
    "PROVIDERS",
    "ensure_model_ready",
    "env_file_is_gitignored",
    "fetch_models_with_fallback",
    "format_llm_error",
    "is_local_model",
    "local_api_base",
    "missing_key_env",
    "model_supports_temperature",
    "persist_to_env_file",
    "provider_for_model",
    "run_setup_flow",
    "select_local_model",
    "select_provider",
    "validate_model",
]
