"""Tests for the interactive shell experience (autointerp_agent.repl).

The full prompt UI is exercised headlessly with prompt_toolkit's pipe-input
test harness; banner/state/observer/dispatch logic is tested directly.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

from rich.console import Console

from autointerp_agent import repl
from autointerp_agent.config import AgentConfig
from autointerp_agent.context import ContextManager
from autointerp_agent.skills import SkillRegistry


def _console(width: int = 100) -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=width)


def _ui(console: Console | None = None, **kwargs) -> repl.SessionUI:
    config = kwargs.pop("config", AgentConfig())
    registry = kwargs.pop("registry", SkillRegistry({}))
    context = kwargs.pop(
        "context", ContextManager(skill_registry=registry, model_name=config.model_name)
    )
    return repl.SessionUI(
        console=console or _console(),
        config=config,
        context=context,
        registry=registry,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# banner + per-user state
# ---------------------------------------------------------------------------


def test_banner_shows_wordmark_model_and_key_state(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    console = _console()
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    repl.print_banner(console, config=config, registry=SkillRegistry({}))
    out = console.file.getvalue()
    assert "█" in out  # the wordmark rendered
    assert "anthropic/claude-sonnet-4-5" in out
    assert "setup will run" in out  # key missing → badge says so
    assert "/help" in out

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    console2 = _console()
    repl.print_banner(console2, config=config, registry=SkillRegistry({}), first_run=True)
    out2 = console2.file.getvalue()
    assert "key configured" in out2
    assert "Welcome" in out2  # first-run greeting


def test_first_run_state_round_trip(tmp_path: Path) -> None:
    assert repl.is_first_run(tmp_path)
    repl.mark_first_run_complete(tmp_path)
    assert not repl.is_first_run(tmp_path)
    state = repl.load_user_state(tmp_path)
    assert "first_run_completed_at" in state
    # Idempotent: a second call keeps the original timestamp.
    first = state["first_run_completed_at"]
    repl.mark_first_run_complete(tmp_path)
    assert repl.load_user_state(tmp_path)["first_run_completed_at"] == first


def test_load_user_state_tolerates_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{not json")
    assert repl.load_user_state(tmp_path) == {}


# ---------------------------------------------------------------------------
# activity indicators
# ---------------------------------------------------------------------------


def test_args_preview_compacts_and_truncates() -> None:
    assert repl._args_preview({"command": "ls -la"}) == "ls -la"
    assert "name" in repl._args_preview({"name": "logit_lens"})
    long = repl._args_preview({"command": "x" * 200})
    assert len(long) <= 71 and long.endswith("…")
    assert repl._args_preview({}) == ""


class _FakeStatus:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def update(self, text: str) -> None:
        self.events.append(f"update:{text}")


def test_observer_prints_tool_lines_around_spinner() -> None:
    console = _console()
    status = _FakeStatus()
    obs = repl.ReplObserver(console, status)
    obs.on_tool_call(0, 0, "id", "update_spec", {"question": "how?"}, "ok", True)
    obs.on_tool_call(0, 1, "id", "bash", {"command": "ls"}, "boom", False)
    out = console.file.getvalue()
    assert "update_spec" in out and "✓" in out
    assert "bash" in out and "✗" in out
    # spinner paused and resumed around each print
    assert status.events.count("stop") == 2 and status.events.count("start") == 2


def test_observer_render_reports_elapsed_and_step() -> None:
    obs = repl.ReplObserver(_console(), _FakeStatus())
    obs._t0 -= 5  # pretend the turn started 5s ago
    obs.iteration = 2
    text = obs._render()
    assert "5s" in text and "step 3" in text


def test_fmt_elapsed_humanizes_durations() -> None:
    assert repl._fmt_elapsed(0) == "0s"
    assert repl._fmt_elapsed(59) == "59s"
    assert repl._fmt_elapsed(68) == "1min 8s"
    assert repl._fmt_elapsed(600) == "10min 0s"
    assert repl._fmt_elapsed(3725) == "1h 2min"


def test_call_llm_resolves_lazily_off_loop(monkeypatch) -> None:
    """The litellm import happens in a worker thread on first use and the
    resolved function is cached for every later call."""
    import autointerp_agent.agent_loop as al

    async def fake_ac(**kwargs):
        return {"ok": kwargs["model"]}

    monkeypatch.setattr(al, "_import_llm_runtime", lambda: fake_ac)
    monkeypatch.setattr(al, "acompletion", None)
    out = asyncio.run(al._call_llm(model="m"))
    assert out == {"ok": "m"}
    assert al.acompletion is fake_ac  # cached — no second import


def test_warm_llm_runtime_idempotent(monkeypatch) -> None:
    import autointerp_agent.agent_loop as al

    started: list[int] = []

    def fake_import():
        started.append(1)
        return "AC"

    monkeypatch.setattr(al, "_import_llm_runtime", fake_import)
    monkeypatch.setattr(al, "acompletion", None)
    monkeypatch.setattr(al, "_warm_thread", None)
    al.warm_llm_runtime()
    assert al._warm_thread is not None
    al._warm_thread.join(timeout=5)
    assert al.acompletion == "AC" and started == [1]
    al.warm_llm_runtime()  # already resolved → no second import
    assert started == [1]


def test_repl_session_warms_llm_runtime(tmp_path: Path, monkeypatch) -> None:
    called: dict = {}
    monkeypatch.setattr(repl, "warm_llm_runtime", lambda: called.setdefault("warm", True))
    result, _, _ = _run_repl_with_keys("/exit\r", tmp_path)
    assert result is None and called.get("warm")


def test_status_timer_ticks_during_slow_turn(monkeypatch) -> None:
    """A single long LLM call must not freeze the elapsed timer — the
    ticker re-renders every second even with zero iteration events."""
    calls = {"n": 0}
    real_refresh = repl.ReplObserver.refresh_status

    def counting(self) -> None:
        calls["n"] += 1
        real_refresh(self)

    monkeypatch.setattr(repl.ReplObserver, "refresh_status", counting)

    async def slow_turn(prompt, config, context, router, observer=None, **kwargs):
        await asyncio.sleep(2.3)
        return "done"

    monkeypatch.setattr(repl, "run_agent_turn", slow_turn)
    out = asyncio.run(repl._execute_turn(_ui(), "hi", object()))
    assert out == "done"
    assert calls["n"] >= 2  # ticked at ~0s, ~1s, ~2s


def test_observer_interim_assistant_text_only_with_tool_calls() -> None:
    console = _console()
    obs = repl.ReplObserver(console, _FakeStatus())
    obs.on_assistant(0, {"content": "final answer", "tool_calls": None})
    assert console.file.getvalue() == ""  # final text is rendered elsewhere
    obs.on_assistant(0, {"content": "let me check the schema", "tool_calls": [{}]})
    assert "let me check the schema" in console.file.getvalue()


# ---------------------------------------------------------------------------
# cost surfacing
# ---------------------------------------------------------------------------


class _StubUsage:
    prompt_tokens = 1000
    completion_tokens = 200
    prompt_tokens_details = None
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _StubResponse:
    model = "claude-sonnet-4-5"
    usage = _StubUsage()
    _hidden_params = {"response_cost": 0.0123}


def test_toolbar_shows_tokens_and_cost() -> None:
    ui = _ui()
    assert "↑0 ↓0" in ui.toolbar_text() and "$0.0000" in ui.toolbar_text()
    # Unlimited by default: a bare turn counter, no "/N" cap suffix.
    assert "turn 0 ·" in ui.toolbar_text()
    ui.cost.add_response(_StubResponse())
    bar = ui.toolbar_text()
    assert "↑1.0k" in bar and "↓200" in bar and "$0.0123" in bar


def test_toolbar_renders_as_box_bottom_border() -> None:
    ui = _ui()
    fragments = ui.toolbar()
    joined = "".join(text for _, text in fragments)
    assert joined.startswith("╰─") and joined.endswith("╯")
    assert "autointerp" in joined


def test_fmt_tokens_scales() -> None:
    assert repl._fmt_tokens(999) == "999"
    assert repl._fmt_tokens(12_300) == "12.3k"
    assert repl._fmt_tokens(2_500_000) == "2.5M"


def test_dispatch_cost_panel() -> None:
    console = _console()
    ui = _ui(console)
    ui.cost.add_response(_StubResponse())
    handled, should_exit = asyncio.run(repl.handle_repl_command("/cost", ui))
    assert handled and not should_exit
    out = console.file.getvalue()
    assert "Total cost" in out and "claude-sonnet-4-5" in out


# ---------------------------------------------------------------------------
# slash-command dispatch
# ---------------------------------------------------------------------------


def test_dispatch_plain_prompt_not_handled() -> None:
    handled, should_exit = asyncio.run(repl.handle_repl_command("how does IOI work?", _ui()))
    assert not handled and not should_exit


def test_dispatch_exit_quit() -> None:
    for cmd in ("/exit", "/quit"):
        handled, should_exit = asyncio.run(repl.handle_repl_command(cmd, _ui()))
        assert handled and should_exit


def test_dispatch_help_and_status_and_skills() -> None:
    console = _console()
    ui = _ui(console)
    for cmd, expect in (("/help", "/model"), ("/status", "api key"), ("/skills", "")):
        handled, should_exit = asyncio.run(repl.handle_repl_command(cmd, ui))
        assert handled and not should_exit
        assert expect in console.file.getvalue()


def test_dispatch_clear_empties_context() -> None:
    ui = _ui()
    ui.context.add_user("hello")
    assert ui.context.messages
    handled, _ = asyncio.run(repl.handle_repl_command("/clear", ui))
    assert handled and ui.context.messages == []


def test_dispatch_unknown_command() -> None:
    console = _console()
    handled, should_exit = asyncio.run(repl.handle_repl_command("/frobnicate", _ui(console)))
    assert handled and not should_exit
    assert "Unknown command" in console.file.getvalue()


def test_dispatch_litrev_topic_priority(monkeypatch) -> None:
    from autointerp_agent import litrev

    calls: list[str] = []

    async def fake_run(topic, **kwargs):
        calls.append(topic)
        return None

    monkeypatch.setattr(litrev, "run_litrev", fake_run)
    monkeypatch.setattr(litrev, "draft_spec_question", lambda *a: None)

    ui = _ui()
    # Explicit argument wins.
    handled, _ = asyncio.run(repl.handle_repl_command("/litrev sparse autoencoders", ui))
    assert handled and calls == ["sparse autoencoders"]
    # Bare command falls back to the last research question.
    ui.last_question = "How do models represent surprise?"
    asyncio.run(repl.handle_repl_command("/litrev", ui))
    assert calls[-1] == "How do models represent surprise?"


def test_dispatch_litrev_without_topic_explains(monkeypatch) -> None:
    from autointerp_agent import litrev

    monkeypatch.setattr(litrev, "draft_spec_question", lambda *a: None)
    console = _console()
    handled, should_exit = asyncio.run(repl.handle_repl_command("/litrev", _ui(console)))
    assert handled and not should_exit
    assert "No topic yet" in console.file.getvalue()


def test_dispatch_litrev_failure_keeps_shell(monkeypatch) -> None:
    from autointerp_agent import litrev

    async def boom(topic, **kwargs):
        raise RuntimeError("arxiv unreachable")

    monkeypatch.setattr(litrev, "run_litrev", boom)
    monkeypatch.setattr(litrev, "draft_spec_question", lambda *a: None)
    console = _console()
    handled, _ = asyncio.run(repl.handle_repl_command("/litrev x", _ui(console)))
    assert handled and "litrev failed" in console.file.getvalue()


def test_dispatch_model_updates_session(monkeypatch) -> None:
    new_config = AgentConfig(model_name="openai/gpt-9")

    async def fake_flow(config, console):
        return new_config

    monkeypatch.setattr(repl, "run_setup_flow", fake_flow)
    ui = _ui()
    handled, should_exit = asyncio.run(repl.handle_repl_command("/model", ui))
    assert handled and not should_exit
    assert ui.config is new_config
    assert ui.context.model_name == "openai/gpt-9"
    assert "openai/gpt-9" in ui.toolbar_text()


def test_dispatch_model_cancel_keeps_session(monkeypatch) -> None:
    async def fake_flow(config, console):
        return None

    monkeypatch.setattr(repl, "run_setup_flow", fake_flow)
    ui = _ui()
    original = ui.config
    handled, _ = asyncio.run(repl.handle_repl_command("/model", ui))
    assert handled and ui.config is original


# ---------------------------------------------------------------------------
# slash-command completion
# ---------------------------------------------------------------------------


def test_slash_completer_suggests_commands() -> None:
    from prompt_toolkit.document import Document

    completer = repl._make_slash_completer(repl.COMMANDS)
    comps = [c.text for c in completer.get_completions(Document("/mo"), None)]
    assert comps == ["/model"]
    comps_all = [c.text for c in completer.get_completions(Document("/"), None)]
    assert "/help" in comps_all and "/exit" in comps_all
    # No completion once a command has arguments or for plain text.
    assert not list(completer.get_completions(Document("/model x"), None))
    assert not list(completer.get_completions(Document("hello"), None))


def _read_boxed(keys: str, history_path: Path) -> str:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    async def go():
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            with create_app_session(input=pipe, output=DummyOutput()):
                return await repl.BoxedInput(_ui(), history_path).read()

    return asyncio.run(go())


def _write_history(path: Path, *entries: str) -> None:
    lines: list[str] = []
    for entry in entries:
        lines += ["", "# 2026-01-01 00:00:00.000000", f"+{entry}"]
    path.write_text("\n".join(lines) + "\n")


def test_boxed_input_widget_builds_and_reads(tmp_path: Path) -> None:
    """The custom input box accepts text and exits on EOF, headlessly."""
    import pytest

    assert _read_boxed("hello world\r", tmp_path / "history") == "hello world"
    # Ctrl-D on an empty buffer raises the EOF exit path.
    with pytest.raises(EOFError):
        _read_boxed("\x04", tmp_path / "history")


def test_boxed_input_tab_accepts_ghost_suggestion(tmp_path: Path) -> None:
    history = tmp_path / "history"
    _write_history(history, "how does GPT-2-small do IOI?")
    # Typing a prefix shows the cross-session ghost; Tab accepts it whole.
    assert _read_boxed("how\t\r", history) == "how does GPT-2-small do IOI?"
    # No suggestion + non-slash text → Tab is a harmless no-op.
    assert _read_boxed("zzz\t\r", history) == "zzz"


def test_boxed_input_up_arrow_recalls_past_session(tmp_path: Path) -> None:
    history = tmp_path / "history"
    _write_history(history, "design the surprise spec")
    assert _read_boxed("\x1b[A\r", history) == "design the surprise spec"


# ---------------------------------------------------------------------------
# the full REPL loop, driven headlessly through pipe input
# ---------------------------------------------------------------------------


def _run_repl_with_keys(keys: str, tmp_path: Path, monkeypatch=None, fake_turn=None,
                        session_store=None, resume_payload=None):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    console = _console()
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    registry = SkillRegistry({})
    context = ContextManager(skill_registry=registry, model_name=config.model_name)
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir(exist_ok=True)

    if fake_turn is not None:
        monkeypatch.setattr(repl, "run_agent_turn", fake_turn)

    async def go():
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            with create_app_session(input=pipe, output=DummyOutput()):
                return await repl.run_spec_repl(
                    config=config,
                    context=context,
                    router=object(),
                    console=console,
                    registry=registry,
                    spec_dir=spec_dir,
                    max_turns=5,
                    state_dir=tmp_path,
                    note="test session",
                    session_store=session_store,
                    resume_payload=resume_payload,
                )

    result = asyncio.run(go())
    _run_repl_with_keys.last_context = context  # type: ignore[attr-defined]
    return result, console.file.getvalue(), spec_dir


def test_repl_exit_command_returns_none(tmp_path: Path) -> None:
    result, output, _ = _run_repl_with_keys("/exit\r", tmp_path)
    assert result is None
    assert "test session" in output  # the caller note rendered
    assert "Ask a research question" in output  # get-started panel
    assert "╭" in output and "╰" in output  # input box opens AND closes


def test_get_started_panel_never_mentions_drafts() -> None:
    # Fresh sessions are genuinely fresh — resumption is explicit via
    # `autointerp --continue`, so the panel must not advertise auto-loading.
    console = _console()
    console.print(repl._get_started_panel())
    out = console.file.getvalue()
    assert "Ask a research question" in out
    assert "previous session" not in out


def test_repl_turn_detects_finalized_spec_and_reports_cost(
    tmp_path: Path, monkeypatch
) -> None:
    async def fake_turn(prompt, config, context, router, observer=None, **kwargs):
        kwargs["cost_tracker"].add_response(_StubResponse())
        spec = tmp_path / "specs" / "demo_rev1.json"
        spec.write_text("{}")
        return "spec **approved**"

    result, output, spec_dir = _run_repl_with_keys(
        "design the IOI spec\r", tmp_path, monkeypatch, fake_turn
    )
    assert result == str(spec_dir / "demo_rev1.json")
    assert "approved" in output  # markdown answer rendered
    assert "⏺ autointerp" in output
    assert "this turn" in output and "$0.0123" in output  # per-turn cost line


def test_cli_import_stays_litellm_free() -> None:
    """Startup-latency guard. `import litellm` reads thousands of files
    (~12s on network filesystems) and must happen on the first agent turn,
    never at CLI import / banner time. Run in a subprocess so this test is
    immune to whatever the pytest process has already imported."""
    import os
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(
        os.environ,
        PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    code = (
        "import sys; import autointerp_agent.cli; "
        "sys.exit(1 if 'litellm' in sys.modules else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env)
    assert proc.returncode == 0, "importing the CLI pulled in litellm"


def _pure_box_tops(output: str) -> int:
    """Count input-box top borders (pure ╭──╮ lines, no panel titles)."""
    return sum(
        1
        for line in output.splitlines()
        if line.strip() and set(line.strip()) <= {"╭", "─", "╮"}
    )


def test_repl_double_ctrl_c_exits_without_box_spam(tmp_path: Path) -> None:
    result, output, _ = _run_repl_with_keys("\x03\x03", tmp_path)
    assert result is None
    # No transcript churn: the armed hint lives in the box's status border,
    # never in scrollback, and nothing prints extra box frames (the live
    # widget renders via the terminal app, not the console transcript).
    assert "use /exit" not in output
    assert _pure_box_tops(output) == 0


def test_repl_single_ctrl_c_clears_input_not_submit(tmp_path: Path, monkeypatch) -> None:
    async def must_not_run(prompt, config, context, router, observer=None, **kwargs):
        raise AssertionError("cleared input must not reach the agent")

    result, output, _ = _run_repl_with_keys(
        "hello\x03/exit\r", tmp_path, monkeypatch, must_not_run
    )
    assert result is None
    assert "AssertionError" not in output


def test_toolbar_shows_exit_hint_only_while_armed() -> None:
    import time as _time

    ui = _ui()
    assert "again to exit" not in ui.toolbar_text()
    ui.ctrl_c_armed_at = _time.monotonic()
    assert "again to exit" in ui.toolbar_text()
    ui.ctrl_c_armed_at = _time.monotonic() - 10  # window expired
    assert "again to exit" not in ui.toolbar_text()


def test_repl_agent_error_keeps_session_alive(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}

    async def fake_turn(prompt, config, context, router, observer=None, **kwargs):
        calls["n"] += 1
        raise RuntimeError("provider exploded")

    result, output, _ = _run_repl_with_keys(
        "hello\r/exit\r", tmp_path, monkeypatch, fake_turn
    )
    assert result is None
    assert calls["n"] == 1
    assert "RuntimeError" in output  # printed, not raised


# ---------------------------------------------------------------------------
# session persistence (fresh by default, --continue restores)
# ---------------------------------------------------------------------------


def test_repl_fresh_session_archives_stray_draft(tmp_path: Path) -> None:
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    (spec_dir / "_draft.json").write_text('{"question": "old plan"}')
    result, output, spec_dir = _run_repl_with_keys("/exit\r", tmp_path)
    assert result is None
    assert not (spec_dir / "_draft.json").exists()
    assert list(spec_dir.glob("_draft-*.bak.json"))
    assert "archived" in output and "--continue" in output


def test_repl_turn_persists_session(tmp_path: Path, monkeypatch) -> None:
    from autointerp_agent.sessions import SessionStore

    async def fake_turn(prompt, config, context, router, observer=None, **kwargs):
        context.add_user(prompt)
        context.add_assistant({"role": "assistant", "content": "noted"})
        return "noted"

    store = SessionStore(cwd=Path("/proj/x"), root=tmp_path / "sessions")
    _run_repl_with_keys(
        "design the IOI spec\r/exit\r", tmp_path, monkeypatch, fake_turn,
        session_store=store,
    )
    metas = store.list()
    assert len(metas) == 1
    payload = store.load(metas[0].session_id)
    assert payload["title"] == "design the IOI spec"
    assert payload["turn"] == 1
    assert any(m.get("role") == "assistant" for m in payload["messages"])


def test_repl_resume_restores_conversation_and_draft(tmp_path: Path) -> None:
    payload = {
        "session_id": "20260101-000000-aaaa",
        "title": "surprise representation",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:10:00Z",
        "model_name": "openai/gpt-5-nano",
        "turn": 3,
        "messages": [
            {"role": "user", "content": "How do models represent surprise?"},
            {"role": "assistant", "content": "Let's design the plan."},
        ],
        "cost": {"total_cost_usd": 0.42, "total_requests": 6, "by_model": {}},
        "draft": '{"question": "surprise?"}',
    }
    result, output, spec_dir = _run_repl_with_keys(
        "/status\r/exit\r", tmp_path, resume_payload=payload,
    )
    assert result is None
    assert "session resumed" in output
    assert "surprise representation" in output
    assert "$0.4200" in output  # restored cost visible in /status
    # The conversation context was restored verbatim for the agent.
    context = _run_repl_with_keys.last_context
    assert context.messages[0]["content"] == "How do models represent surprise?"
    # The draft plan travels with its session.
    assert (spec_dir / "_draft.json").read_text() == '{"question": "surprise?"}'
    # No get-started panel on resume.
    assert "Ask a research question" not in output


def test_dispatch_clear_rotates_session(tmp_path: Path) -> None:
    from autointerp_agent.sessions import SessionStore

    ui = _ui()
    ui.session_store = SessionStore(cwd=Path("/p"), root=tmp_path)
    ui.session_id = "old-id"
    ui.turn = 4
    ui.spec_dir = tmp_path
    handled, _ = asyncio.run(repl.handle_repl_command("/clear", ui))
    assert handled
    assert ui.session_id != "old-id" and ui.turn == 0 and not ui.resumed
