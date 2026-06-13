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
    missing_key_env,
    run_setup_flow,
)
from .sessions import (
    archive_stray_draft,
    cost_from_dict,
    restore_draft,
    snapshot_draft,
)
from .skills import SkillRegistry

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
    key_badge = (
        "[green]✓ key configured[/green]"
        if key_missing is None
        else f"[yellow]✗ no key — setup will run ({key_missing})[/yellow]"
    )

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
    key_line = (
        "[green]✓ configured[/green]" if env_var is None
        else f"[yellow]✗ missing ({env_var})[/yellow]"
    )
    draft = Path("outputs/specs/_draft.json")
    tokens_in, tokens_out = _usage_totals(ui.cost)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column()
    table.add_row("model", ui.config.model_name)
    table.add_row("api key", key_line)
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


class ReplObserver:
    """TurnObserver that narrates the agent's work in the shell.

    One dim line per tool call (Claude-Code style), interim assistant
    thoughts in italics, and a status line re-rendered every second by the
    ticker in ``_execute_turn`` (a single LLM call can run for 30s+, so
    updating only at iteration boundaries would freeze the elapsed timer).
    """

    def __init__(self, console: Console, status: Any) -> None:
        self.console = console
        self.status = status
        self.iteration = 0
        self._t0 = time.monotonic()

    def _render(self) -> str:
        elapsed = _fmt_elapsed(int(time.monotonic() - self._t0))
        return (
            f"[bold cyan]✶[/bold cyan] [dim]working… {elapsed} · "
            f"step {self.iteration + 1} · ctrl-c to interrupt[/dim]"
        )

    def refresh_status(self) -> None:
        self.status.update(self._render())

    def _println(self, text: str) -> None:
        self.status.stop()
        self.console.print(text)
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
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        preview = _args_preview(args)
        suffix = f"[dim]({preview})[/dim]" if preview else ""
        self._println(f"  {mark} [bold]{name}[/bold]{suffix}")

    def on_final(self, iteration: int, final_text: str) -> None:
        pass


# ---------------------------------------------------------------------------
# The prompt (input box, completion, history, toolbar)
# ---------------------------------------------------------------------------


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
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.layout.processors import AppendAutoSuggestion
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
        input_window = Window(
            BufferControl(
                buffer=buffer, input_processors=[AppendAutoSuggestion()]
            ),
            get_line_prefix=_prefix,
            wrap_lines=True,
            dont_extend_height=True,
        )
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
            refresh_interval=0.5,  # lets the armed-exit hint expire visibly
        )

    async def read(self) -> str:
        """Show the box and return the submitted text (EOFError on exit)."""
        self.buffer.reset()
        self.buffer.load_history_if_not_yet_loaded()  # no-op once hydrated
        return await self.app.run_async()


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

    messages = payload.get("messages") or []
    last_user = next(
        (m.get("content") for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        "",
    )
    last_assistant = next(
        (m.get("content") for m in reversed(messages)
         if m.get("role") == "assistant" and isinstance(m.get("content"), str)
         and m.get("content")),
        "",
    )
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim", justify="right", no_wrap=True)
    info.add_column()
    info.add_row("topic", _trunc(str(payload.get("title") or "(untitled)"), 70))
    info.add_row("started", iso_age_string(str(payload.get("created_at") or "")))
    info.add_row("turns", str(payload.get("turn") or 0))
    info.add_row("model then", str(payload.get("model_name") or "?"))
    info.add_row("draft plan", "restored" if payload.get("draft") else "none")
    body: list[Any] = [info]
    if last_user or last_assistant:
        body += [
            Text(""),
            Text.from_markup(f"[bold cyan]❯[/bold cyan] [dim]{_trunc(last_user)}[/dim]"),
            Text.from_markup(f"[bold cyan]⏺[/bold cyan] [dim]{_trunc(last_assistant)}[/dim]"),
        ]
    return Panel(
        Group(*body), box=rich_box.ROUNDED, border_style="green",
        title="session resumed", title_align="left", padding=(0, 1),
    )


def _persist_session(ui: SessionUI) -> None:
    """Write the session after each turn. Persistence never crashes the shell."""
    if ui.session_store is None or ui.session_id is None or ui.turn == 0:
        return
    title = next(
        (m.get("content") for m in ui.context.messages
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        "(untitled)",
    )
    try:
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


def render_answer(console: Console, answer: str) -> None:
    # Markdown pulls in markdown_it (~0.5s on network filesystems) — defer
    # it past startup; the first call lands after an LLM round-trip anyway.
    from rich.markdown import Markdown

    console.print()
    console.print("[bold cyan]⏺ autointerp[/bold cyan]")
    console.print(Padding(Markdown(answer or "*(no answer)*"), (0, 0, 0, 2)))
    console.print()


# ---------------------------------------------------------------------------
# The shared Stage-0 REPL loop
# ---------------------------------------------------------------------------


def _snapshot_specs(spec_dir: Path) -> set[Path]:
    if not spec_dir.exists():
        return set()
    return {p for p in spec_dir.glob("*_rev*.json") if not p.name.startswith("_")}


async def _tick_status(observer: ReplObserver) -> None:
    """Re-render the working… status every second so the timer counts up."""
    while True:
        observer.refresh_status()
        await asyncio.sleep(1.0)


async def _execute_turn(
    ui: SessionUI, prompt: str, router: Any
) -> str | None:
    """One agent turn with spinner + live tool feed. None on interrupt/error."""
    status = ui.console.status(
        "[bold cyan]✶[/bold cyan] [dim]thinking…[/dim]", spinner="dots"
    )
    observer = ReplObserver(ui.console, status)
    status.start()
    ticker = asyncio.create_task(_tick_status(observer))
    try:
        return await run_agent_turn(
            prompt, ui.config, ui.context, router,
            observer=observer, cost_tracker=ui.cost,
        )
    except KeyboardInterrupt:
        ui.console.print(
            "[yellow]⏸ interrupted — turn aborted, conversation context kept.[/yellow]"
        )
        return None
    except Exception as exc:  # noqa: BLE001 — keep the shell alive
        ui.console.print(f"[red]{format_llm_error(exc)}[/red]")
        return None
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
        status.stop()


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
) -> Optional[str]:
    """Run the interactive Stage-0 shell until a spec is finalized or exit.

    Returns the path of the finalized spec (a new ``*_rev*.json`` appearing
    in ``spec_dir``), or None on exit / turn-cap. Fresh sessions never
    inherit prior state (stray drafts get archived); ``resume_payload``
    (from ``--continue``) restores conversation, draft, turns, and cost.
    """
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
                return None
            ui.ctrl_c_armed_at = None
            prompt = prompt.strip()
            if not prompt:
                continue
            handled, should_exit = await handle_repl_command(prompt, ui)
            if should_exit:
                return None
            if handled:
                continue

        ui.turn += 1
        ui.last_question = prompt
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
        if new_specs:
            return str(sorted(new_specs)[-1])
        if cap and ui.turn >= cap:
            console.print(
                f"[yellow]Turn cap reached ({ui.turn}/{cap}) without an approved "
                f"spec. Re-run with `--max-turns N` to extend.[/yellow]"
            )
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
