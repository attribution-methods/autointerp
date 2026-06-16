"""Interactive shell experience for the autointerp CLIs.

Everything that makes ``autointerp`` feel like a real terminal app lives
here: the welcome banner, persistent per-user state (``~/.autointerp/``),
a Claude-Code-style prompt (slash-command completion popup, cross-session
history, bottom status toolbar, bordered input), live tool-call indicators
with a spinner while the agent works, Ctrl-C interrupting the current turn
instead of the session, and markdown-rendered answers.

Both Stage-0 REPL loops (``cli.py`` bare ``autointerp`` and ``app.py``
``autointerp investigate``) run through :func:`run_spec_repl` so the shell
behaves identically everywhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from rich import box as rich_box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autointerp.utils.cost import CostTracker

from .agent_loop import run_agent_turn, warm_llm_runtime
from .config import DEFAULT_MODEL, USER_DIR, AgentConfig
from .context import ContextManager
from .model_select import (
    clear_credentials,
    format_llm_error,
    is_local_model,
    local_api_base,
    missing_key_env,
    model_supports_temperature,
    run_setup_flow,
)
from .sessions import (
    archive_stray_draft,
    cost_from_dict,
    restore_draft,
    snapshot_draft,
)
from .skills import SkillRegistry
from .stage0_tools import set_approval_gate, set_user_approved

# ---------------------------------------------------------------------------
# Per-user state (~/.autointerp/)
# ---------------------------------------------------------------------------


def load_user_state(state_dir: Path | None = None) -> dict[str, Any]:
    path = (state_dir or USER_DIR) / "state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_user_state(state: dict[str, Any], state_dir: Path | None = None) -> Path:
    directory = state_dir or USER_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    path.write_text(json.dumps(state, indent=2) + "\n")
    return path


def is_first_run(state_dir: Path | None = None) -> bool:
    return "first_run_completed_at" not in load_user_state(state_dir)


def mark_first_run_complete(state_dir: Path | None = None) -> None:
    state = load_user_state(state_dir)
    if "first_run_completed_at" in state:
        return
    state["first_run_completed_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    save_user_state(state, state_dir)


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------

_WORDMARK = (
    "▄▀█ █░█ ▀█▀ █▀█ █ █▄░█ ▀█▀ █▀▀ █▀█ █▀█",
    "█▀█ █▄█ ░█░ █▄█ █ █░▀█ ░█░ ██▄ █▀▄ █▀░",
)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("autointerp")
    except Exception:  # noqa: BLE001 — not installed / editable without metadata
        return "dev"


def _short_path(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def print_banner(
    console: Console,
    *,
    config: AgentConfig,
    registry: SkillRegistry,
    first_run: bool = False,
) -> None:
    """The launch screen: wordmark, session facts, and command hints."""
    key_missing = missing_key_env(config.model_name)
    if is_local_model(config.model_name):
        key_badge = f"[cyan]local · {local_api_base()}[/cyan]"
    elif key_missing is None:
        key_badge = "[green]✓ key configured[/green]"
    else:
        key_badge = f"[yellow]✗ no key — setup will run ({key_missing})[/yellow]"

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim", justify="right", no_wrap=True)
    info.add_column()
    info.add_row("model", f"[bold]{config.model_name}[/bold]   {key_badge}")
    info.add_row("skills", f"{len(registry.skills)} method skills loaded")
    info.add_row("cwd", _short_path(Path.cwd()))

    body = Group(
        Text(_WORDMARK[0], style="bold cyan"),
        Text(_WORDMARK[1], style="cyan"),
        Text("agentic mechanistic interpretability", style="dim italic"),
        Text(""),
        info,
        Text(""),
        Text("/help commands · /model switch model · Ctrl-D exit", style="dim"),
    )
    console.print(
        Panel(
            body,
            box=rich_box.ROUNDED,
            border_style="cyan",
            padding=(1, 3),
            subtitle=f"[dim]v{_version()}[/dim]",
            subtitle_align="right",
            expand=False,
        )
    )
    if first_run:
        console.print(
            "[bold]Welcome![/bold] [dim]First run — your model, API key, and history "
            "will be saved (.env + ~/.autointerp) so this only happens once.[/dim]"
        )


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

COMMANDS: dict[str, str] = {
    "/model": "switch model / provider / API key (lists models live, validates)",
    "/logout": "sign out — forget the saved API key + model (setup runs next launch)",
    "/temperature": "set sampling temperature (0.0–2.0), or 'default'. "
                    "Usage: /temperature [value]",
    "/litrev": "arXiv literature panel for a topic (side channel — never "
               "enters the agent's context). Usage: /litrev [topic]",
    "/cost": "token usage and cost, per model, for this session",
    "/status": "session status — model, key, turns, tokens, cost",
    "/skills": "list the available method skills",
    "/clear": "clear conversation context and screen",
    "/help": "show available commands",
    "/exit": "leave the session (also /quit or Ctrl-D)",
}


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _usage_totals(tracker: CostTracker) -> tuple[int, int]:
    """(input incl. cache, output) token totals across all models."""
    tokens_in = sum(
        mu.input_tokens + mu.cache_read_input_tokens + mu.cache_creation_input_tokens
        for mu in tracker.by_model.values()
    )
    tokens_out = sum(mu.output_tokens for mu in tracker.by_model.values())
    return tokens_in, tokens_out


def _fmt_cost(tracker: CostTracker) -> str:
    suffix = "+?" if tracker.has_unknown_cost else ""
    return f"${tracker.total_cost_usd:.4f}{suffix}"


@dataclass
class SessionUI:
    """Mutable session state shared by the prompt, toolbar, and commands."""

    console: Console
    config: AgentConfig
    context: ContextManager
    registry: SkillRegistry
    max_turns: int | None = None
    turn: int = 0
    note: str | None = None  # one-line caller note shown under the banner
    extra_commands: dict[str, str] = field(default_factory=dict)
    cost: CostTracker = field(default_factory=lambda: CostTracker(run_id="repl-session"))
    # Double-Ctrl-C exit arming (monotonic timestamp of the first press).
    ctrl_c_armed_at: float | None = None
    # Most recent non-slash user prompt — bare /litrev derives its topic here.
    last_question: str | None = None
    # Session persistence (set by run_spec_repl when a store is attached).
    session_store: Any = None
    session_id: str | None = None
    resumed: bool = False
    spec_dir: Path | None = None

    def toolbar_text(self) -> str:
        if (
            self.ctrl_c_armed_at is not None
            and time.monotonic() - self.ctrl_c_armed_at < _CTRL_C_EXIT_WINDOW
        ):
            return "press Ctrl-C again to exit · any key to stay"
        cap = f"/{self.max_turns}" if self.max_turns else ""
        tokens_in, tokens_out = _usage_totals(self.cost)
        return (
            f"autointerp · {self.config.model_name} · turn {self.turn}{cap} · "
            f"↑{_fmt_tokens(tokens_in)} ↓{_fmt_tokens(tokens_out)} · "
            f"{_fmt_cost(self.cost)} · /help"
        )

    def toolbar(self) -> list[tuple[str, str]]:
        """Render the toolbar as the input box's live bottom border.

        prompt_toolkit draws the bottom toolbar directly beneath the input
        line in non-fullscreen prompts, so styling it as ``╰─ status ─╯``
        closes the box while the user types; the plain ``╰───╯`` printed on
        submit replaces it in scrollback.
        """
        inner = f" {self.toolbar_text()} "
        width = max(20, self.console.width)
        pad = max(0, width - 2 - len(inner) - 1)
        return [
            ("class:prompt.border", "╰─"),
            ("class:toolbar.text", inner),
            ("class:prompt.border", "─" * pad + "╯"),
        ]


_CTRL_C_EXIT_WINDOW = 2.0  # seconds for the second Ctrl-C to confirm exit


def _help_panel(ui: SessionUI) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="dim")
    for cmd, desc in {**COMMANDS, **ui.extra_commands}.items():
        table.add_row(cmd, desc)
    table.add_row("", "")
    table.add_row("[dim]anything else[/dim]", "sent to the agent")
    return Panel(table, box=rich_box.ROUNDED, border_style="dim", title="commands",
                 title_align="left", expand=False)


def _status_panel(ui: SessionUI) -> Panel:
    env_var = missing_key_env(ui.config.model_name)
    if is_local_model(ui.config.model_name):
        key_line = f"[cyan]local · {local_api_base()}[/cyan]"
    elif env_var is None:
        key_line = "[green]✓ configured[/green]"
    else:
        key_line = f"[yellow]✗ missing ({env_var})[/yellow]"
    draft = Path("outputs/specs/_draft.json")
    tokens_in, tokens_out = _usage_totals(ui.cost)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column()
    table.add_row("model", ui.config.model_name)
    table.add_row("api key", key_line)
    table.add_row("temperature", _temperature_status(ui.config))
    table.add_row("turns", f"{ui.turn}/{ui.max_turns or '∞'}")
    table.add_row("context", f"{len(ui.context.messages)} messages")
    table.add_row("tokens", f"↑{_fmt_tokens(tokens_in)} ↓{_fmt_tokens(tokens_out)}")
    table.add_row("cost", f"{_fmt_cost(ui.cost)} ({ui.cost.total_requests} requests)")
    table.add_row("skills", str(len(ui.registry.skills)))
    table.add_row("draft spec", "in progress" if draft.exists() else "none")
    if ui.session_id:
        table.add_row(
            "session",
            f"{ui.session_id} ({'resumed' if ui.resumed else 'fresh'})",
        )
    return Panel(table, box=rich_box.ROUNDED, border_style="dim", title="status",
                 title_align="left", expand=False)


def _flush_screen(console: Console) -> None:
    """Wipe the visible screen *and* the scrollback buffer for a clean slate."""
    console.clear()
    with contextlib.suppress(Exception):
        console.file.write("\x1b[3J")  # also drop scrollback, not just the screen
        console.file.flush()


def _fresh_screen(ui: SessionUI) -> None:
    """Flush to a clean screen, then redraw the banner — plus the get-started
    intro when there's no conversation yet (e.g. after /clear or a re-login)."""
    _flush_screen(ui.console)
    print_banner(ui.console, config=ui.config, registry=ui.registry, first_run=False)
    if not ui.context.messages:
        ui.console.print(_get_started_panel())


def _temperature_status(config: AgentConfig) -> str:
    """One-line temperature state for the status panel."""
    if config.temperature is None:
        return "provider default"
    if not model_supports_temperature(config.model_name):
        return f"{config.temperature:g} [yellow](ignored — reasoning model)[/yellow]"
    return f"{config.temperature:g}"


def _handle_temperature(prompt: str, ui: SessionUI) -> None:
    """``/temperature [value|default]`` — show or set the agent's sampling
    temperature. None (the default) omits the param so the provider's own
    default applies; OpenAI reasoning models ignore any value set here."""
    parts = prompt.split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else ""
    model = ui.config.model_name
    if not arg:  # show
        current = "provider default (1.0 for OpenAI/Anthropic)" \
            if ui.config.temperature is None else f"{ui.config.temperature:g}"
        body = f"temperature: [bold]{current}[/bold]"
        if not model_supports_temperature(model):
            body += (f"\n[yellow]note:[/yellow] {model} is a reasoning model — "
                     "it only accepts its default, so a custom value is ignored.")
        body += "\n[dim]set with /temperature <0.0–2.0>, reset with /temperature default[/dim]"
        ui.console.print(Panel(body, box=rich_box.ROUNDED, border_style="dim",
                               title="temperature", title_align="left", expand=False))
        return
    if arg in ("default", "none", "auto", "reset", "unset"):
        ui.config = ui.config.model_copy(update={"temperature": None})
        ui.console.print("[dim]temperature → provider default[/dim]")
        return
    try:
        value = float(arg)
    except ValueError:
        ui.console.print("[yellow]Usage: /temperature <number 0.0–2.0>, or "
                         "/temperature default[/yellow]")
        return
    if not 0.0 <= value <= 2.0:
        ui.console.print("[yellow]Temperature must be between 0.0 and 2.0.[/yellow]")
        return
    ui.config = ui.config.model_copy(update={"temperature": value})
    msg = f"temperature → [bold]{value:g}[/bold]"
    if not model_supports_temperature(model):
        msg += (f"  [yellow](note: {model} is a reasoning model and ignores "
                "temperature — this takes effect if you switch to a standard "
                "model)[/yellow]")
    ui.console.print(f"[dim]{msg}[/dim]")


async def handle_repl_command(prompt: str, ui: SessionUI) -> tuple[bool, bool]:
    """Dispatch ``/...`` commands. Returns ``(handled, should_exit)``.

    ``/model`` mutates ``ui.config`` / ``ui.context`` in place so the toolbar
    and subsequent turns pick up the change.
    """
    if not prompt.startswith("/"):
        return False, False
    command = prompt.split()[0].lower()
    if command in ("/exit", "/quit"):
        return True, True
    if command == "/help":
        ui.console.print(_help_panel(ui))
        return True, False
    if command == "/status":
        ui.console.print(_status_panel(ui))
        return True, False
    if command == "/cost":
        ui.console.print(
            Panel(ui.cost.format_summary(), box=rich_box.ROUNDED, border_style="dim",
                  title="session cost", title_align="left", expand=False)
        )
        return True, False
    if command == "/skills":
        for line in ui.registry.list_lines():
            ui.console.print(f"[dim]{line}[/dim]")
        return True, False
    if command == "/clear":
        ui.context.messages.clear()
        if ui.session_store is not None:
            # A cleared conversation is a new session (the old one stays on
            # disk at its last saved turn, still resumable via --continue).
            ui.session_id = ui.session_store.new_id()
            ui.resumed = False
            ui.turn = 0
            if ui.spec_dir is not None:
                archive_stray_draft(ui.spec_dir)
        _fresh_screen(ui)
        ui.console.print("[dim]context cleared — new session[/dim]")
        return True, False
    if command == "/model":
        updated = await run_setup_flow(ui.config, ui.console)
        if updated is None:
            ui.console.print(f"[dim]kept {ui.config.model_name}[/dim]")
        else:
            ui.config = updated
            ui.context.model_name = updated.model_name
            # Fresh screen + banner + intro so a re-login lands on a clean
            # slate reflecting the new model.
            _fresh_screen(ui)
        return True, False
    if command == "/logout":
        cleared = clear_credentials()
        for path, names in cleared.items():
            ui.console.print(f"[dim]removed {', '.join(names)} from {path}[/dim]")
        if not cleared:
            ui.console.print("[dim]no saved credentials on disk[/dim]")
        ui.config = ui.config.model_copy(update={"model_name": DEFAULT_MODEL})
        ui.context.model_name = DEFAULT_MODEL
        ui.console.print(
            "[yellow]Signed out.[/yellow] [dim]Run /model to sign in again, "
            "or /exit and relaunch.[/dim]"
        )
        return True, False
    if command == "/temperature":
        _handle_temperature(prompt, ui)
        return True, False
    if command == "/litrev":
        from . import litrev

        parts = prompt.split(maxsplit=1)
        topic = (
            (parts[1].strip() if len(parts) > 1 else None)
            or litrev.draft_spec_question()
            or ui.last_question
        )
        if not topic:
            ui.console.print(
                "[yellow]No topic yet — ask a research question first, or "
                "pass one: /litrev <topic>.[/yellow]"
            )
            return True, False
        try:
            await litrev.run_litrev(
                topic,
                console=ui.console,
                model_name=ui.config.model_name,
                cost_tracker=ui.cost,
            )
        except Exception as exc:  # noqa: BLE001 — a failed search must not kill the shell
            ui.console.print(f"[red]litrev failed: {format_llm_error(exc)}[/red]")
        return True, False
    ui.console.print(f"[yellow]Unknown command {command!r} — try /help.[/yellow]")
    return True, False


# ---------------------------------------------------------------------------
# Live activity indicators (spinner + tool-call feed)
# ---------------------------------------------------------------------------


def _fmt_elapsed(seconds: int) -> str:
    """Human duration: 45s · 1min 8s · 1h 2min."""
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}min {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}min"


def _args_preview(args: dict[str, Any]) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    if "command" in args:
        text = str(args["command"])
    else:
        try:
            text = json.dumps(args, default=str).strip("{}")
        except (TypeError, ValueError):
            text = str(args)
    text = " ".join(text.split())
    return text[:70] + ("…" if len(text) > 70 else "")


# Successful calls to these render nothing: they are the agent reading
# documentation/state, which informs the agent, not the user. The spinner
# already shows liveness; failures still always render (with the reason).
_SILENT_TOOLS = frozenset({
    "describe_spec", "show_spec", "validate_spec", "list_metrics",
    "read_metric", "list_skills", "read_skill", "read_file",
    "get_state", "get_budget", "current_stage", "get_progress",
})


# Schema class names that may appear inside tool ERROR text. The agent needs
# the precise originals to self-correct; the user-facing feed substitutes.
_REASON_SUBS = (
    ("InvestigationSpec", "plan"),
    ("PartialSpec", "draft plan"),
    ("StageSpec", "stage"),
    ("BehaviorSpec", "behavior"),
)

# Failure verb phrases per tool, so ✗ lines speak the same plain language as
# ✓ lines. Unknown tools fall back to their raw name (debuggable, rare).
_FAIL_LABELS = {
    "update_spec": "update the plan",
    "remove_spec_fields": "edit the plan",
    "finalize_spec": "finalize the plan",
    "validate_spec": "validate the plan",
    "describe_spec": "read the plan schema",
    "show_spec": "render the plan",
    "propose_custom_metric": "add the custom metric",
    "list_metrics": "list the metrics",
    "read_metric": "read the metric reference",
    "list_skills": "list the skills",
    "read_skill": "read the skill",
    "bash": "run bash",
    "read_file": "read the file",
    "write_file": "write the file",
    "edit_file": "edit the file",
    "compute_metric": "compute the metric",
    "compute_and_commit_metric": "compute the metric",
    "commit_artifact": "commit the artifact",
    "evaluate_criterion": "evaluate the criterion",
    "advance_stage": "advance the stage",
    "request_spec_revision": "request a revision",
}


def _humanize_field(key: str) -> str:
    """Internal field name → plain words (``success_criteria`` → ``success
    criteria``). Keeps the user's vocabulary free of code identifiers."""
    return str(key).replace("_", " ")


def _display_path(run_dir: Any, path: Any) -> str:
    """Render a tool's file path as its REAL location. During an investigation
    the agent's relative paths resolve inside the run dir, so `scripts/x.py` is
    shown as `runs/<id>/scripts/x.py` — never confused with the repo's own
    scripts/."""
    p = str(path or "?")
    if not run_dir or Path(p).is_absolute():
        return p
    full = Path(run_dir) / p
    try:
        return full.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return full.as_posix()


def _one_line_cmd(command: str) -> str:
    """Render a multi-line bash script as one legible line: distinct commands
    joined with ' ; ' rather than flattened into a run-on (newline→space made
    `ls -la\\nls dir` read as the single bogus command `ls -la ls dir`)."""
    return " ; ".join(ln.strip() for ln in command.splitlines() if ln.strip())


def _format_tool_event(
    name: str, args: dict[str, Any], ok: bool, output: str = "",
    run_dir: Any = None,
) -> str | None:
    """One human line per tool call, or None to suppress.

    Design rule: a line must either tell the user what just changed in
    their world (plan edits, files, approvals, metrics) or surface a
    problem with its reason. Raw (name, json-args) dumps do neither.
    """
    args = args if isinstance(args, dict) else {}
    if not ok:
        reason = output.removeprefix("ERROR: ").strip().splitlines()
        detail = _trunc(reason[0], 90) if reason and reason[0] else "failed"
        for internal, plain in _REASON_SUBS:
            detail = detail.replace(internal, plain)
        label = _FAIL_LABELS.get(name)
        head = f"couldn't {label}" if label else name
        return f"  [red]✗[/red] [bold]{head}[/bold][dim] — {detail}[/dim]"
    if name in _SILENT_TOOLS:
        return None
    if name == "update_spec":
        keys = list((args.get("patch") or {}).keys())
        what = ", ".join(_humanize_field(k) for k in keys[:4]) + (
            "…" if len(keys) > 4 else ""
        )
        suffix = f"[dim] · {what}[/dim]" if what else ""
        return f"  [green]✓[/green] [bold]updated plan[/bold]{suffix}"
    if name == "remove_spec_fields":
        keys = ", ".join(_humanize_field(str(k)) for k in (args.get("keys") or [])[:4])
        return f"  [green]✓[/green] [bold]removed from plan[/bold][dim] · {keys}[/dim]"
    if name == "finalize_spec":
        return "  [green]✓[/green] [bold]plan approved and locked in[/bold]"
    if name == "bash":
        return (
            f"  [green]✓[/green] [bold]bash[/bold]"
            f"[dim] · {_trunc(_one_line_cmd(str(args.get('command') or '')), 70)}[/dim]"
        )
    if name in ("write_file", "edit_file"):
        verb = "wrote" if name == "write_file" else "edited"
        return (
            f"  [green]✓[/green] [bold]{verb}[/bold]"
            f"[dim] · {_display_path(run_dir, args.get('path'))}[/dim]"
        )
    if name == "plan":
        return "  [green]✓[/green] [bold]updated plan[/bold]"
    if name in ("compute_metric", "compute_and_commit_metric"):
        metric = args.get("metric") or args.get("name") or "metric"
        return f"  [green]✓[/green] [bold]computed {metric}[/bold]"
    if name == "commit_artifact":
        kind = str(args.get("kind", "artifact"))
        return f"  [green]✓[/green] [bold]committed {_HUMANIZE.get(kind, kind)}[/bold]"
    if name == "evaluate_criterion":
        return (
            f"  [green]✓[/green] [bold]evaluated criterion[/bold]"
            f"[dim] · {args.get('criterion_id', '?')}[/dim]"
        )
    if name == "advance_stage":
        return "  [green]✓[/green] [bold]advanced to next stage[/bold]"
    if name == "request_spec_revision":
        return "  [green]✓[/green] [bold]requested spec revision[/bold]"
    if name == "propose_custom_metric":
        return (
            f"  [green]✓[/green] [bold]proposed custom metric[/bold]"
            f"[dim] · {args.get('name', '?')}[/dim]"
        )
    # Unknown / MCP tools: keep the generic style rather than hiding them.
    preview = _args_preview(args)
    suffix = f"[dim]({preview})[/dim]" if preview else ""
    return f"  [green]✓[/green] [bold]{name}[/bold]{suffix}"


# Consecutive tool failures with no intervening success before we stop
# treating them as transient self-correction and surface them as ✗ lines.
_FAILURE_FLUSH_THRESHOLD = 3


class ReplObserver:
    """TurnObserver that narrates the agent's work in the shell.

    One dim line per tool call (Claude-Code style), interim assistant
    thoughts in italics, and a status line re-rendered every second by the
    ticker in ``_execute_turn`` (a single LLM call can run for 30s+, so
    updating only at iteration boundaries would freeze the elapsed timer).
    """

    def __init__(self, console: Console, status: Any, run_dir: Any = None) -> None:
        self.console = console
        self.status = status
        self.run_dir = run_dir
        self.iteration = 0
        self._t0 = time.monotonic()
        # Transient tool failures the agent may still recover from. Held back
        # from the feed and shown only as a spinner note; surfaced as ✗ lines
        # only if the agent can't recover (too many in a row, or at turn end).
        self._pending_failures: list[str] = []

    def _render(self) -> str:
        elapsed = _fmt_elapsed(int(time.monotonic() - self._t0))
        note = ""
        if self._pending_failures:
            note = f" · self-correcting ({len(self._pending_failures)})"
        return (
            f"[bold cyan]✶[/bold cyan] [dim]working… {elapsed} · "
            f"step {self.iteration + 1} · ctrl-c to interrupt{note}[/dim]"
        )

    def refresh_status(self) -> None:
        self.status.update(self._render())

    def _println(self, text: str) -> None:
        self.status.stop()
        self.console.print(text)
        self.status.start()

    def flush_pending_failures(self) -> None:
        """Surface held-back failures as ✗ lines (the agent didn't recover).

        Consecutive identical failures collapse to one line with a ``×N``
        count, so a genuinely stuck agent (e.g. repeating the same malformed
        metric call) reads as one clear problem rather than a wall.
        """
        if not self._pending_failures:
            return
        pending, self._pending_failures = self._pending_failures, []
        self.status.stop()
        i = 0
        while i < len(pending):
            j = i
            while j + 1 < len(pending) and pending[j + 1] == pending[i]:
                j += 1
            count = j - i + 1
            suffix = f" [dim](×{count})[/dim]" if count > 1 else ""
            self.console.print(pending[i] + suffix)
            i = j + 1
        self.status.start()

    def on_iteration_start(self, iteration: int) -> None:
        self.iteration = iteration
        self.refresh_status()

    def on_assistant(self, iteration: int, message: dict[str, Any]) -> None:
        content = str(message.get("content") or "").strip()
        if content and message.get("tool_calls"):
            snippet = " ".join(content.split())[:160]
            self._println(f"  [dim italic]{snippet}[/dim italic]")

    def on_tool_call(
        self,
        iteration: int,
        call_idx: int,
        call_id: str,
        name: str,
        args: dict[str, Any],
        output: str,
        ok: bool,
    ) -> None:
        line = _format_tool_event(name, args, ok, output, run_dir=self.run_dir)
        if ok:
            # Any success means the agent worked past its transient failures.
            # Drop them (the spinner already showed "self-correcting").
            if self._pending_failures:
                self._pending_failures = []
                self.refresh_status()
            if line is not None:
                self._println(line)
            return
        # A failure: hold it. The agent usually fixes its own mistake on the
        # next call (gate errors are how it learns the schema), so don't alarm
        # the user yet — show it only if recovery doesn't come.
        if line is not None:
            self._pending_failures.append(line)
        if len(self._pending_failures) >= _FAILURE_FLUSH_THRESHOLD:
            self.flush_pending_failures()
        else:
            self.refresh_status()

    def on_final(self, iteration: int, final_text: str) -> None:
        # Turn ended — any still-pending failures were never recovered.
        self.flush_pending_failures()


# ---------------------------------------------------------------------------
# The prompt (input box, completion, history, toolbar)
# ---------------------------------------------------------------------------


def _word_wrap(text: str, width: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Greedy word wrap with Claude-Code semantics.

    Returns ``(rows, pos)`` where ``pos[i]`` is the (row, col) of character
    ``i`` and ``pos[len(text)]`` is the end-of-text cursor cell. The space at
    a break point is consumed (never rendered at the start of the next row)
    and words are kept whole unless longer than the width.
    """
    width = max(1, width)
    text = text.replace("\n", " ")
    rows: list[str] = [""]
    pos: list[tuple[int, int]] = []
    row, col = 0, 0

    def _newline() -> None:
        nonlocal row, col
        rows.append("")
        row += 1
        col = 0

    for token in re.findall(r"\S+|\s+", text):
        if token.isspace():
            for _ in token:
                if col >= width:
                    # The breaking space is swallowed visually; its cursor
                    # cell aliases the start of the next row.
                    _newline()
                    pos.append((row, col))
                else:
                    pos.append((row, col))
                    rows[row] += " "
                    col += 1
            continue
        if col > 0 and col + len(token) > width and len(token) <= width:
            # Word doesn't fit — wrap it whole. A single trailing space
            # already rendered stays where it is (end of the prior row).
            _newline()
        for ch in token:
            if col >= width:
                _newline()  # word longer than the row: hard split
            pos.append((row, col))
            rows[row] += ch
            col += 1
    if col >= width:
        # Text exactly fills the row: the end-of-text cursor lives at the
        # start of a fresh row (a real cell), never one past the right edge.
        _newline()
    pos.append((row, col))
    return rows, pos


def _make_slash_completer(commands: dict[str, str]) -> Any:
    """Completer for ``/`` commands. Defined lazily so importing this module
    never pays the prompt_toolkit import (startup-latency budget)."""
    from prompt_toolkit.completion import Completer, Completion

    class _SlashCompleter(Completer):
        def get_completions(self, document, complete_event) -> Iterable[Completion]:
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:
                return
            for cmd, desc in commands.items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd, start_position=-len(text), display=cmd, display_meta=desc
                    )

    return _SlashCompleter()


class BoxedInput:
    """Codex/Claude-style input widget: a live four-sided box.

    Dynamic height — one line when empty, grows with wrapped text, and
    expands only while the completion menu is open (the menu renders inside
    the box, above the bottom border). The bottom border doubles as the
    status bar (model · turn · tokens · cost). The whole frame persists in
    scrollback after submit.

    Built as a custom prompt_toolkit Application because PromptSession
    cannot place a toolbar directly beneath the input: it reserves
    ``reserve_space_for_menu`` rows and pins the toolbar below them, which
    reads as a giant empty box.
    """

    def __init__(self, ui: SessionUI, history_path: Path) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.controls import (
            BufferControl,
            FormattedTextControl,
            UIContent,
        )
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.styles import Style

        self.ui = ui
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = Buffer(
            history=FileHistory(str(history_path)),
            completer=_make_slash_completer({**COMMANDS, **ui.extra_commands}),
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            multiline=False,
        )
        buffer = self.buffer
        # Hydrate past sessions synchronously so ghost suggestions and
        # ↑-recall work from the first keystroke. load_history_strings() is
        # the documented sync source; priming the loaded cache touches
        # internals, so any failure falls back to the official background
        # load scheduled in read().
        history = buffer.history
        try:
            history._loaded_strings = list(history.load_history_strings())
            history._loaded = True
        except Exception:  # noqa: BLE001 — degrade to async load
            pass

        def _prefix(line_number: int, wrap_count: int):
            if line_number == 0 and wrap_count == 0:
                return [("class:prompt.border", "│ "), ("class:prompt.caret", "❯ ")]
            return [("class:prompt.border", "│ "), ("", "  ")]

        kb = KeyBindings()

        @kb.add("enter")
        def _accept(event) -> None:
            # If the completion menu is open, accept the highlighted item — or
            # the first one when nothing is explicitly selected. The menu auto-
            # opens via complete_while_typing without selecting anything, so a
            # bare Enter would otherwise submit the half-typed command. Apply
            # only when it would change the text, so a fully-typed command
            # still submits on the first Enter.
            state = buffer.complete_state
            if state is not None:
                comp = state.current_completion
                if comp is None:
                    completions = getattr(state, "completions", None) or []
                    comp = completions[0] if completions else None
                if comp is not None and comp.text != buffer.text:
                    buffer.apply_completion(comp)
                    return
            text = buffer.text
            if text.strip():
                buffer.history.append_string(text)
            event.app.exit(result=text)

        @kb.add("c-i")  # tab
        def _complete(event) -> None:
            # Priority: cycle an open menu → accept the ghost suggestion →
            # open the slash-command menu. Tab-accepts-suggestion is the
            # convention users expect (Claude Code does the same).
            if buffer.complete_state:
                buffer.complete_next()
                return
            suggestion = buffer.suggestion
            if suggestion is None and buffer.auto_suggest is not None and buffer.text:
                # The ghost is computed by a background task; on a fast Tab
                # right after a keystroke it may not have run yet — compute
                # synchronously so Tab always honors what would be shown.
                suggestion = buffer.auto_suggest.get_suggestion(buffer, buffer.document)
            if suggestion is not None and suggestion.text:
                buffer.insert_text(suggestion.text)
                return
            if buffer.text.startswith("/"):
                buffer.start_completion(select_first=True)

        @kb.add("up")
        def _up(event) -> None:
            if buffer.complete_state:
                buffer.complete_previous()
            else:
                buffer.history_backward()

        @kb.add("down")
        def _down(event) -> None:
            if buffer.complete_state:
                buffer.complete_next()
            else:
                buffer.history_forward()

        @kb.add("escape", eager=True)
        def _dismiss(event) -> None:
            buffer.cancel_completion()

        @kb.add(
            "right",
            filter=Condition(
                lambda: buffer.suggestion is not None
                and buffer.cursor_position == len(buffer.text)
            ),
        )
        @kb.add(
            "end",
            filter=Condition(
                lambda: buffer.suggestion is not None
                and buffer.cursor_position == len(buffer.text)
            ),
        )
        def _take_suggestion(event) -> None:
            if buffer.suggestion:
                buffer.insert_text(buffer.suggestion.text)

        # Ctrl-C inside the editor: with text → clear; empty → arm the
        # status-border hint; second press within the window exits.
        @kb.add("c-c")
        def _ctrl_c(event) -> None:
            if buffer.text:
                buffer.reset()
                ui.ctrl_c_armed_at = None
                return
            now = time.monotonic()
            if (
                ui.ctrl_c_armed_at is not None
                and now - ui.ctrl_c_armed_at < _CTRL_C_EXIT_WINDOW
            ):
                ui.ctrl_c_armed_at = None
                event.app.exit(exception=EOFError())
            else:
                ui.ctrl_c_armed_at = now
                # The first press is otherwise silent — tell the user how to
                # actually leave. Guarded so it can never break the key binding.
                try:
                    from prompt_toolkit.application import run_in_terminal

                    run_in_terminal(
                        lambda: ui.console.print(
                            "[dim](press Ctrl-C again or Ctrl-D to exit)[/dim]"
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass

        @kb.add("c-d", filter=Condition(lambda: not buffer.text))
        def _eof(event) -> None:
            event.app.exit(exception=EOFError())

        top = VSplit(
            [
                Window(width=1, height=1, char="╭", style="class:prompt.border"),
                Window(height=1, char="─", style="class:prompt.border"),
                Window(width=1, height=1, char="╮", style="class:prompt.border"),
            ],
            height=1,
        )
        _PREFIX_WIDTH = 4  # "│ ❯ " / "│   " rendered by _prefix

        class _WordWrapControl(BufferControl):
            """BufferControl that emits pre-word-wrapped rows.

            prompt_toolkit's Window only character-wraps, which strands the
            breaking space at the start of continuation rows and splits
            words mid-token. Editing, focus, and key handling stay on the
            real Buffer; only the rendered UIContent (and the ghost
            suggestion, drawn manually) is custom.
            """

            def _layout_rows(self, width: int):
                text = buffer.text
                ghost = ""
                if (
                    buffer.suggestion
                    and buffer.suggestion.text
                    and buffer.cursor_position == len(text)
                ):
                    ghost = buffer.suggestion.text
                full = text + ghost
                rows, pos = _word_wrap(full, max(1, width - _PREFIX_WIDTH))
                fragment_rows: list[list[tuple[str, str]]] = [[] for _ in rows]
                for i, ch in enumerate(full):
                    if ch == " " and pos[i] == pos[i + 1]:
                        continue  # space consumed at the wrap point
                    style = "" if i < len(text) else "class:auto-suggestion"
                    fragment_rows[pos[i][0]].append((style, ch))
                for row_fragments in fragment_rows:
                    # The screen cursor can only land on cells that were
                    # actually written; an invisible trailing space makes the
                    # end-of-row cell real (otherwise prompt_toolkit's lookup
                    # misses and parks the cursor at the window origin).
                    row_fragments.append(("", " "))
                cursor = pos[min(buffer.cursor_position, len(full))]
                return fragment_rows, Point(x=cursor[1], y=cursor[0])

            def preferred_height(
                self, width, max_available_height, wrap_lines, get_line_prefix
            ):
                fragment_rows, _ = self._layout_rows(width)
                return len(fragment_rows)

            def create_content(self, width, height, preview_search=False):
                fragment_rows, cursor = self._layout_rows(width)
                return UIContent(
                    get_line=lambda row: fragment_rows[row],
                    line_count=len(fragment_rows),
                    cursor_position=cursor,
                    show_cursor=True,
                )

        input_window = Window(
            _WordWrapControl(buffer=buffer),
            get_line_prefix=_prefix,
            wrap_lines=False,  # rows arrive pre-wrapped at word boundaries
            dont_extend_height=True,
        )
        self._input_window = input_window  # exposed for rendering tests
        body = VSplit(
            [
                input_window,
                Window(width=1, char="│", style="class:prompt.border"),
            ]
        )

        def _pad_row() -> VSplit:
            # One blank bordered row of interior padding above/below the text.
            return VSplit(
                [
                    Window(width=1, height=1, char="│", style="class:prompt.border"),
                    Window(height=1, char=" "),
                    Window(width=1, height=1, char="│", style="class:prompt.border"),
                ],
                height=1,
            )

        bottom = Window(
            FormattedTextControl(ui.toolbar), height=1, dont_extend_height=True
        )
        root = HSplit(
            [
                top,
                _pad_row(),
                body,
                _pad_row(),
                CompletionsMenu(max_height=6, scroll_offset=0),
                bottom,
            ]
        )
        self.app: Any = Application(
            layout=Layout(root, focused_element=input_window),
            key_bindings=kb,
            style=Style.from_dict(
                {
                    "prompt.border": "ansibrightblack",
                    "prompt.caret": "bold ansicyan",
                    "toolbar.text": "ansicyan",
                    "completion-menu.completion": "bg:ansiblack ansiwhite",
                    "completion-menu.completion.current": "bg:ansicyan ansiblack",
                    "completion-menu.meta.completion": "bg:ansiblack ansibrightblack",
                    "completion-menu.meta.completion.current": "bg:ansicyan ansiblack",
                }
            ),
            full_screen=False,
            mouse_support=False,
            erase_when_done=False,  # the submitted box stays in scrollback
            # No refresh_interval: an inline app redraws on each keypress,
            # which is all we need. A periodic refresh re-draws the box while
            # the user sits idle, and those redraws nest instead of updating
            # in place (one stacked frame per tick). The only cost of dropping
            # it: the Ctrl-C "press again to exit" toolbar hint clears on the
            # next keypress rather than auto-expiring after the 2s window.
        )

    async def read(self) -> str:
        """Show the box and return the submitted text (EOFError on exit).

        ``set_exception_handler=False``: prompt_toolkit otherwise installs a
        handler that, for *any* loop event without an exception (e.g. a
        background litellm/httpx cleanup task GC'd mid-render), prints
        "Unhandled exception in event loop … Press ENTER to continue" and
        freezes the UI. We keep the session's own quiet handler instead.
        """
        self.buffer.reset()
        self.buffer.load_history_if_not_yet_loaded()  # no-op once hydrated
        return await self.app.run_async(set_exception_handler=False)


def _input_border_top(console: Console) -> None:
    width = max(20, console.width)
    console.print(f"[bright_black]╭{'─' * (width - 2)}╮[/bright_black]")


def _input_border_bottom(console: Console) -> None:
    width = max(20, console.width)
    console.print(f"[bright_black]╰{'─' * (width - 2)}╯[/bright_black]")


def _echo_scripted_prompt(console: Console, prompt: str) -> None:
    """Render a non-typed (CLI-arg) prompt exactly like a typed one."""
    _input_border_top(console)
    console.print(f"[bright_black]│[/bright_black] [bold cyan]❯[/bold cyan] {prompt}")
    _input_border_bottom(console)


def _trunc(text: str, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text[: n - 1] + "…" if len(text) > n else text


def _resumed_panel(payload: dict[str, Any]) -> Panel:
    from autointerp.utils.age import iso_age_string

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim", justify="right", no_wrap=True)
    info.add_column()
    info.add_row("topic", _trunc(str(payload.get("title") or "(untitled)"), 70))
    info.add_row("started", iso_age_string(str(payload.get("created_at") or "")))
    info.add_row("turns", str(payload.get("turn") or 0))
    info.add_row("model then", str(payload.get("model_name") or "?"))
    info.add_row("draft plan", "restored" if payload.get("draft") else "none")
    return Panel(
        info, box=rich_box.ROUNDED, border_style="green",
        title="session resumed", title_align="left", padding=(0, 1),
    )


def _replay_transcript(console: Console, messages: list[dict[str, Any]]) -> None:
    """Re-render a restored conversation in the live session's own visual
    language (the Claude-Code resume experience): user prompts as input
    boxes, assistant answers as markdown, tool calls as the ✓/✗ activity
    feed. Tool outputs stay collapsed — the mark carries their outcome.
    """
    tool_results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            tool_results[str(msg.get("tool_call_id"))] = str(msg.get("content") or "")

    console.rule("[dim]previous conversation[/dim]", style="bright_black")
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            if isinstance(content, str) and content.strip():
                _echo_scripted_prompt(console, content.strip())
            continue
        if role != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        if isinstance(content, str) and content.strip() and calls:
            snippet = " ".join(content.split())[:160]
            console.print(f"  [dim italic]{snippet}[/dim italic]")
        for call in calls:
            function = (call or {}).get("function") or {}
            name = str(function.get("name") or "?")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            output = tool_results.get(str((call or {}).get("id")), "")
            ok = not output.startswith("ERROR: ")
            line = _format_tool_event(
                name, args if isinstance(args, dict) else {}, ok, output
            )
            if line is not None:
                console.print(line)
        if isinstance(content, str) and content.strip() and not calls:
            render_answer(console, content)
    console.rule("[dim]session restored — continue below[/dim]", style="bright_black")


def _persist_session(ui: SessionUI, force: bool = False) -> None:
    """Write the session after each turn. Persistence never crashes the shell.

    ``force`` saves even at turn 0 (used on exit) so the resume command shown on
    the way out always resolves to a loadable session."""
    if ui.session_store is None or ui.session_id is None or (ui.turn == 0 and not force):
        return
    try:
        title = next(
            (m.get("content") for m in ui.context.messages
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            "(untitled)",
        )
        ui.session_store.save(
            ui.session_id,
            model_name=ui.config.model_name,
            messages=ui.context.messages,
            turn=ui.turn,
            cost=ui.cost.to_dict(),
            title=_trunc(str(title), 80),
            draft=snapshot_draft(ui.spec_dir) if ui.spec_dir else None,
        )
    except Exception:  # noqa: BLE001
        pass


def _get_started_panel() -> Panel:
    lines: list[Text] = [
        Text("Ask a research question about a model behavior, in plain English:"),
        Text.from_markup(
            '  [cyan]"How does GPT-2-small do indirect object identification?"[/cyan]'
        ),
        Text.from_markup(
            '  [cyan]"Where does Qwen2.5-7B-Instruct represent refusal?"[/cyan]'
        ),
        Text(""),
        Text.from_markup(
            "Together we'll design the [bold]investigation plan[/bold] — which "
            "model to study, what data and metrics to use, and what result would "
            "count as success or failure. Once you approve the plan, the "
            "investigation runs on its own and writes its findings to runs/."
        ),
    ]
    return Panel(
        Group(*lines), box=rich_box.ROUNDED, border_style="dim",
        title="get started", title_align="left", padding=(0, 1),
    )


# snake_case domain identifiers (>=1 underscore): metric / tool / field names
# the agent writes as bare words. Wrapped in backticks so the markdown
# renderer styles them (theme markdown.code = cyan) and scanning a wall of
# plan text is easy. Intraword underscores are not markdown emphasis, so this
# is also safer than leaving them bare.
_SNAKE_IDENT = re.compile(r"(?<![\w`])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w`])")
# Code regions (already styled) we must not touch: fenced blocks + inline code.
_CODE_REGION = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)


def _highlight_identifiers(markdown_text: str) -> str:
    """Backtick-wrap bare snake_case identifiers, skipping existing code."""
    out: list[str] = []
    last = 0
    for region in _CODE_REGION.finditer(markdown_text):
        out.append(_SNAKE_IDENT.sub(r"`\1`", markdown_text[last:region.start()]))
        out.append(region.group(0))  # leave code spans/fences untouched
        last = region.end()
    out.append(_SNAKE_IDENT.sub(r"`\1`", markdown_text[last:]))
    return "".join(out)


# Internal CamelCase schema/type names the planner sometimes leaks as nouns in
# prose. We translate them to plain language deterministically — a backstop to
# the plain-language retry, which can't catch identifier-ish names the model
# *invents* (the user's "InvestigationPlan" is not even a real schema name).
# Only prose is rewritten; genuine code spans/fences are left intact.
_HUMANIZE = {
    "InvestigationSpec": "investigation plan",
    "PartialSpec": "draft plan",
    "InvestigationStage": "stage",
    "StageSpec": "stage",
    "BehaviorSpec": "behavior",
    "ModelRef": "model",
    "DatasetSpec": "dataset",
    "ContrastSpec": "contrast set",
    "CustomMetricDef": "custom metric",
    "SuccessCriterion": "success criterion",
    "Criterion": "success criterion",
    "PromptBatch": "prompt batch",
    "ActivationCacheRef": "activation cache",
    # typed-artifact names the planner leaks when narrating Stage-2/3 work
    "InterventionResult": "intervention result",
    "MetricResult": "metric result",
    "ValidationResult": "validation result",
    "GenerationSample": "generation sample",
    "FeatureFinding": "feature finding",
    "BehavioralFinding": "behavioral finding",
    "CandidateSite": "candidate site",
}
# Distinctive terms are COLORED like the underscored identifiers (rendered as
# inline code, which the session theme paints accent-cyan). Everyday single
# words are humanized but left plain — coloring only the schema-derived
# instances of "model"/"stage" would clash with the same word used plainly.
_HUMANIZE_PLAIN = {"model", "dataset", "stage", "behavior", "success criterion"}

# A SINGLE pass catches every form the planner leaks "investigation plan" /
# typed-schema names in, so a replacement is never re-scanned (no double-wrap):
#   - ident:  exact CamelCase identifiers from the map (InterventionResult, …)
#   - camel:  any Investigation<Capital…> token, incl. invented "InvestigationPlan"
#   - phrase: the spaced/Title/lower phrase "Investigation Plan" / "investigation
#             plan" (the form a real run leaked — Investigation[A-Z]\w* missed it
#             because of the space). The capital-or-space guards keep the plain
#             word "Investigations" from ever matching.
_LEAK_RE = re.compile(
    r"\b(?P<ident>" + "|".join(map(re.escape, _HUMANIZE)) + r")\b"
    r"|\b(?P<camel>Investigation[A-Z]\w*)\b"
    r"|\b(?P<phrase>(?i:investigation[ \t-]+plan))\b"
)


def _humanized(phrase: str) -> str:
    return phrase if phrase in _HUMANIZE_PLAIN else f"`{phrase}`"


def _leak_sub(m: "re.Match[str]") -> str:
    ident = m.group("ident")
    if ident is not None:
        return _humanized(_HUMANIZE[ident])
    return "`investigation plan`"  # camel/phrase → the colored canonical term


def _humanize_segment(prose: str) -> str:
    return _LEAK_RE.sub(_leak_sub, prose)


def _humanize_identifiers(markdown_text: str) -> str:
    """Replace internal schema/type identifiers with plain language in PROSE,
    leaving code spans/fences untouched so genuine ``code`` references survive."""
    out: list[str] = []
    last = 0
    for region in _CODE_REGION.finditer(markdown_text):
        out.append(_humanize_segment(markdown_text[last:region.start()]))
        out.append(region.group(0))
        last = region.end()
    out.append(_humanize_segment(markdown_text[last:]))
    return "".join(out)


def render_answer(console: Console, answer: str) -> None:
    # Markdown pulls in markdown_it (~0.5s on network filesystems) — defer
    # it past startup; the first call lands after an LLM round-trip anyway.
    from rich.markdown import Markdown
    from rich.theme import Theme

    console.print()
    console.print("[bold cyan]⏺ autointerp[/bold cyan]")
    md = Markdown(_highlight_identifiers(_humanize_identifiers(answer or "*(no answer)*")))
    # Colour inline code (now incl. the highlighted identifiers) in the
    # session's cyan accent, without disturbing fenced-block syntax colours.
    with console.use_theme(Theme({"markdown.code": "cyan"})):
        console.print(Padding(md, (0, 0, 0, 2)))
    console.print()


# ---------------------------------------------------------------------------
# The shared Stage-0 REPL loop
# ---------------------------------------------------------------------------


def _snapshot_specs(spec_dir: Path) -> set[Path]:
    if not spec_dir.exists():
        return set()
    return {p for p in spec_dir.glob("*_rev*.json") if not p.name.startswith("_")}


_APPROVAL_RE = re.compile(
    r"^(?:i\s+)?(?:approve|approved|yes|yep|yeah|ok|okay|sure|"
    r"go(?:\s+ahead|\s+on|\s+for\s+it)?|continue|keep\s+going|proceed|"
    r"lock\s+it(?:\s+in)?|do\s+it|finali[sz]e|run\s+it|launch(?:\s+it)?|"
    r"ship\s+it|let'?s\s+go|sounds?\s+good|lgtm|confirm(?:ed)?|looks?\s+good)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(but|however|except|instead|change|don'?t|not|wait)\b", re.I)
# An unambiguous approval PHRASE anywhere in a longer message — so answering the
# agent's questions and approving in one breath ("I want a local model … I
# approve otherwise.") still counts. Kept explicit so it won't fire on chatter.
_APPROVAL_PHRASE = re.compile(
    r"\b(i\s+approve|approved|go\s+ahead|looks?\s+good|lgtm|sounds?\s+good|"
    r"ship\s+it|i'?m\s+good\s+with\s+(?:it|this|that))\b",
    re.IGNORECASE,
)


def _print_resume_hint(console: Console, ui: "SessionUI") -> None:
    """On exit, print the exact command to resume THIS session (Codex/CC style)."""
    sid = getattr(ui, "session_id", None)
    if getattr(ui, "session_store", None) is None or not sid:
        return
    _persist_session(ui, force=True)  # ensure `--continue <id>` resolves on exit
    # highlight=False stops rich's auto-highlighter from tinting only the DIGIT
    # runs of the id cyan (leaving the hex suffix + dashes white) — the whole
    # command renders in one uniform accent colour instead.
    console.print(
        f"[dim]Resume this session with[/dim] "
        f"[bold cyan]autointerp --continue {sid}[/bold cyan]",
        highlight=False,
    )


def _looks_like_approval(text: str) -> bool:
    """A clean approval — a short 'Approve'/'yes go ahead', OR an explicit
    approval phrase inside a longer reply ('… I approve otherwise.'). Never a
    request for changes ('yes, but use a bigger model' / 'approve after you
    change the model')."""
    t = (text or "").strip()
    if not t or _NEGATION_RE.search(t):
        return False
    if len(t.split()) <= 6 and _APPROVAL_RE.match(t):
        return True
    return bool(_APPROVAL_PHRASE.search(t))


# Re-prompt when the user approved but the agent narrated locking the plan in
# without actually calling finalize_spec (a weak-model failure that leaves the
# user unsure whether anything launched).
_FINALIZE_NUDGE = (
    "The user approved the plan, but you did NOT call finalize_spec — so nothing "
    "was locked in and no investigation started. If the draft plan is ready, "
    "call finalize_spec NOW to lock it in and launch the run. Do not describe "
    "locking it or claim it is running — actually call the tool. Do NOT re-print "
    "the plan or ask for approval again. If the plan is genuinely not ready, fix "
    "it, then finalize."
)


async def _force_finalize(
    router: Any, console: Console, spec_dir: Path, before: set[Path]
) -> set[Path]:
    """Finalize deterministically on a clear user approval.

    A reluctant driver (weak model) sometimes re-renders the plan and asks for
    approval again instead of calling finalize_spec, stranding an approved plan.
    Since the human explicitly approved, the REPL locks it in directly via the
    same tool the agent would call (finalize_spec needs no approval). This is a
    no-op if the draft isn't actually finalize-ready — finalize_spec returns its
    blocker, which we surface so the agent can fix it on the next turn — so a
    half-formed plan is never launched. Returns the set of newly written specs."""
    try:
        out, ok = await router.call_tool(
            "finalize_spec", {"approver": "user", "approver_kind": "human"}
        )
    except Exception:  # noqa: BLE001 — never crash the shell on the fallback
        return set()
    if ok:
        return _snapshot_specs(spec_dir) - before
    first = next((ln for ln in (out or "").splitlines() if ln.strip()), "")
    if "structural error" in first.lower():
        # The draft isn't fully filled in — don't dump the pydantic blob; the
        # agent fixes it on the next turn. Plain language for the user.
        console.print(
            "[yellow]The plan isn't fully filled in yet — the agent is still "
            "finishing it. Give it a moment, or say what to change.[/yellow]"
        )
    elif first:
        clean = first.replace("Cannot finalize — ", "").strip()
        console.print(f"[yellow]Not ready to lock in yet: {clean}[/yellow]")
    return set()


async def _tick_status(observer: ReplObserver) -> None:
    """Re-render the working… status every second so the timer counts up."""
    while True:
        observer.refresh_status()
        await asyncio.sleep(1.0)


class _FanoutObserver:
    """Dispatch each TurnObserver hook to several observers at once.

    Used to drive the live console feed and a persistent transcript writer
    from a single turn. A failure in one observer never blocks the others or
    the agent loop (same contract as the loop's own ``_emit``).
    """

    def __init__(self, observers: list[Any]) -> None:
        self._observers = observers

    def _dispatch(self, hook: str, *args: Any) -> None:
        for obs in self._observers:
            fn = getattr(obs, hook, None)
            if fn is None:
                continue
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 — an observer must never break the loop
                pass

    def on_iteration_start(self, *a: Any) -> None:
        self._dispatch("on_iteration_start", *a)

    def on_assistant(self, *a: Any) -> None:
        self._dispatch("on_assistant", *a)

    def on_tool_call(self, *a: Any) -> None:
        self._dispatch("on_tool_call", *a)

    def on_final(self, *a: Any) -> None:
        self._dispatch("on_final", *a)


async def run_live_turn(
    prompt: str,
    config: AgentConfig,
    context: ContextManager,
    router: Any,
    console: Console,
    *,
    cost_tracker: CostTracker | None = None,
    extra_observer: Any | None = None,
    initial_label: str = "thinking…",
) -> str:
    """One agent turn rendered with the live spinner + semantic tool feed.

    Sets up the elapsed-time ticker and the ``ReplObserver`` console feed;
    when ``extra_observer`` is given (e.g. an investigation transcript
    writer) events fan out to both. Exceptions propagate — the caller
    decides how to present an interrupt or a provider error.
    """
    status = console.status(
        f"[bold cyan]✶[/bold cyan] [dim]{initial_label}[/dim]", spinner="dots"
    )
    repl_observer = ReplObserver(console, status, run_dir=getattr(router, "run_dir", None))
    observer: Any = (
        repl_observer
        if extra_observer is None
        else _FanoutObserver([repl_observer, extra_observer])
    )
    status.start()
    ticker = asyncio.create_task(_tick_status(repl_observer))
    try:
        return await run_agent_turn(
            prompt, config, context, router,
            observer=observer, cost_tracker=cost_tracker,
        )
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
        # Surface any unrecovered transient failures (e.g. the turn hit the
        # iteration cap, so on_final never fired).
        repl_observer.flush_pending_failures()
        status.stop()


async def _execute_turn(
    ui: SessionUI, prompt: str, router: Any
) -> str | None:
    """One agent turn with spinner + live tool feed. None on interrupt/error."""
    try:
        return await run_live_turn(
            prompt, ui.config, ui.context, router, ui.console, cost_tracker=ui.cost,
        )
    except KeyboardInterrupt:
        ui.console.print(
            "[yellow]⏸ interrupted — turn aborted, conversation context kept.[/yellow]"
        )
        return None
    except Exception as exc:  # noqa: BLE001 — keep the shell alive
        ui.console.print(f"[red]{format_llm_error(exc)}[/red]")
        return None


def install_quiet_loop_handler() -> None:
    """Drop benign background-task noise so it can't hijack the input UI.

    LLM clients (litellm/httpx) sometimes leave a cleanup task that asyncio
    garbage-collects later — emitting a loop event with no ``exception``
    ("Task was destroyed but it is pending"). With prompt_toolkit's own
    handler that becomes a "Press ENTER to continue" UI freeze; here we
    swallow only those benign events and pass real exceptions through.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    previous = loop.get_exception_handler()

    def handler(loop_: Any, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        message = str(context.get("message", ""))
        if exc is None or "destroyed but it is pending" in message:
            return  # benign — ignore
        if previous is not None:
            previous(loop_, context)
        else:
            loop_.default_exception_handler(context)

    loop.set_exception_handler(handler)


async def run_spec_repl(
    *,
    config: AgentConfig,
    context: ContextManager,
    router: Any,
    console: Console,
    registry: SkillRegistry,
    spec_dir: Path,
    initial_prompt: str | None = None,
    max_turns: int = 0,  # 0 = unlimited (the human at the keyboard is the cap)
    state_dir: Path | None = None,
    note: str | None = None,
    cost_tracker: CostTracker | None = None,
    session_store: Any = None,
    resume_payload: dict[str, Any] | None = None,
    on_finalize: Any = None,
) -> Optional[str]:
    """Run the interactive Stage-0 shell until a spec is finalized or exit.

    Returns the path of the finalized spec (a new ``*_rev*.json`` appearing
    in ``spec_dir``), or None on exit / turn-cap. Fresh sessions never
    inherit prior state (stray drafts get archived); ``resume_payload``
    (from ``--continue``) restores conversation, draft, turns, and cost.

    ``on_finalize``: optional ``async (spec_path) -> None`` callback. When set,
    a finalized spec is handed to it (the investigation runs in-session) and
    the loop continues afterwards instead of returning — so the user stays in
    the conversation. When None, the spec path is returned to the caller.
    """
    install_quiet_loop_handler()
    cap = max_turns if max_turns and max_turns > 0 else None
    ui = SessionUI(
        console=console, config=config, context=context, registry=registry, max_turns=cap
    )
    if cost_tracker is not None:
        ui.cost = cost_tracker
    ui.spec_dir = spec_dir
    ui.session_store = session_store
    if session_store is not None:
        ui.session_id = (
            str(resume_payload["session_id"])
            if resume_payload and resume_payload.get("session_id")
            else session_store.new_id()
        )
    if resume_payload:
        ui.resumed = True
        restored = list(resume_payload.get("messages") or [])
        context.messages.clear()
        context.messages.extend(restored)
        ui.turn = int(resume_payload.get("turn") or 0)
        if cost_tracker is None:
            ui.cost = cost_from_dict(
                resume_payload.get("cost"), ui.session_id or "resumed"
            )
        restore_draft(spec_dir, resume_payload.get("draft"))
        ui.last_question = next(
            (m.get("content") for m in reversed(restored)
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            None,
        )
        console.print(_resumed_panel(resume_payload))
        _replay_transcript(console, restored)
    else:
        archived = archive_stray_draft(spec_dir)
        if archived is not None:
            console.print(
                f"[dim]fresh session — previous unfinished plan archived to "
                f"{archived.name} (resume sessions with `autointerp --continue`)[/dim]"
            )
    box = BoxedInput(ui, (state_dir or USER_DIR) / "history")
    # Import litellm in the background while the user reads/types — the
    # first turn then starts warm instead of stalling the loop (and the
    # elapsed-time ticker) for the duration of the import.
    warm_llm_runtime()
    # Gate finalize on an explicit user go-ahead: the weak driver otherwise
    # drafts AND finalizes on turn 0, launching a plan the user never approved.
    set_approval_gate(True)
    before = _snapshot_specs(spec_dir)
    if note:
        console.print(f"[dim]{note}[/dim]")
    if initial_prompt is None and not resume_payload:
        console.print(_get_started_panel())

    pending = initial_prompt.strip() if initial_prompt else None
    while True:
        if pending:
            prompt, pending = pending, None
            _echo_scripted_prompt(console, prompt)
        else:
            try:
                prompt = await box.read()
            except KeyboardInterrupt:
                # Normally unreachable — Ctrl-C is consumed by the editor
                # key binding. Kept for environments that still deliver
                # SIGINT between prompts.
                console.print("[dim]use /exit, Ctrl-D, or double Ctrl-C to leave[/dim]")
                continue
            except EOFError:
                console.print()
                _print_resume_hint(console, ui)
                return None
            ui.ctrl_c_armed_at = None
            prompt = prompt.strip()
            if not prompt:
                continue
            handled, should_exit = await handle_repl_command(prompt, ui)
            if should_exit:
                _print_resume_hint(console, ui)
                return None
            if handled:
                continue

        ui.turn += 1
        ui.last_question = prompt
        # Tell the finalize gate whether the user approved on THIS turn, so the
        # agent can only lock in a plan the user has actually green-lit.
        set_user_approved(_looks_like_approval(prompt))
        usd_before = ui.cost.total_cost_usd
        in_before, out_before = _usage_totals(ui.cost)
        requests_before = ui.cost.total_requests
        answer = await _execute_turn(ui, prompt, router)
        if answer is not None:
            render_answer(console, answer)
        if ui.cost.total_requests > requests_before:
            tokens_in, tokens_out = _usage_totals(ui.cost)
            console.print(
                f"[dim]↑{_fmt_tokens(tokens_in - in_before)} "
                f"↓{_fmt_tokens(tokens_out - out_before)} · "
                f"${ui.cost.total_cost_usd - usd_before:.4f} this turn · "
                f"{_fmt_cost(ui.cost)} session[/dim]"
            )
        _persist_session(ui)

        new_specs = _snapshot_specs(spec_dir) - before
        # The user approved (e.g. "Continue."), but the agent re-rendered the plan
        # or narrated "locking it in" without actually calling finalize_spec, so
        # nothing launched. Nudge it ONCE to do it; if it still won't, finalize
        # deterministically — the human approved, so the model's reluctance must
        # not strand the run. (_force_finalize is a no-op if the draft isn't ready.)
        if not new_specs and on_finalize is not None and _looks_like_approval(prompt):
            console.print(
                "[dim]…the plan wasn't locked in yet — finalizing and starting "
                "the run…[/dim]"
            )
            nudge_answer = await _execute_turn(ui, _FINALIZE_NUDGE, router)
            if nudge_answer is not None:
                render_answer(console, nudge_answer)
            _persist_session(ui)
            new_specs = _snapshot_specs(spec_dir) - before
            if not new_specs:
                new_specs = await _force_finalize(router, console, spec_dir, before)
            _persist_session(ui)

        if new_specs:
            finalized = str(sorted(new_specs)[-1])
            if on_finalize is None:
                return finalized  # caller drives the investigation (e.g. app.py)
            # Run the investigation in-session, then return to the prompt so the
            # user can read results and keep going (the Codex/CC way). The
            # investigation must NEVER be able to take the session down — an error
            # (or Ctrl-C) is caught here, surfaced briefly, and we stay at the
            # prompt so the user can revise the plan, tweak the metric, or retry.
            try:
                await on_finalize(finalized)
            except KeyboardInterrupt:
                console.print(
                    "[yellow]⏸ investigation interrupted — the session is still "
                    "here.[/yellow]"
                )
            except Exception as exc:  # noqa: BLE001 — keep the session alive
                first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
                console.print(
                    f"[red]The investigation stopped on an error: "
                    f"{_trunc(first, 200)}[/red]"
                )
                console.print(
                    "[dim]The session is still alive — revise the plan, change the "
                    "metric, or start another investigation.[/dim]"
                )
            before = _snapshot_specs(spec_dir)  # don't re-trigger on this spec
            _persist_session(ui)
            console.print(
                "[dim]Back to design — ask a follow-up, start another "
                "investigation, or /exit.[/dim]"
            )
            continue
        if cap and ui.turn >= cap:
            console.print(
                f"[yellow]Turn cap reached ({ui.turn}/{cap}) without an approved "
                f"spec. Re-run with `--max-turns N` to extend.[/yellow]"
            )
            _print_resume_hint(console, ui)
            return None
        if cap and ui.turn == max(1, int(cap * 0.8)):
            console.print(f"[dim]({ui.turn}/{cap} turns used)[/dim]")


__all__ = [
    "COMMANDS",
    "ReplObserver",
    "SessionUI",
    "handle_repl_command",
    "is_first_run",
    "load_user_state",
    "mark_first_run_complete",
    "print_banner",
    "render_answer",
    "run_spec_repl",
    "save_user_state",
]
