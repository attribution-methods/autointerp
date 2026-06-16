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


def test_banner_shows_local_model_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1")
    console = _console()
    config = AgentConfig(model_name="hosted_vllm/Qwen/Qwen2.5-7B-Instruct")
    repl.print_banner(console, config=config, registry=SkillRegistry({}))
    out = console.file.getvalue()
    assert "hosted_vllm/Qwen/Qwen2.5-7B-Instruct" in out
    assert "local" in out and "localhost:8000" in out  # endpoint, not a key badge
    assert "no key" not in out


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


def test_observer_prints_successes_and_suppresses_lookups() -> None:
    console = _console()
    status = _FakeStatus()
    obs = repl.ReplObserver(console, status)
    obs.on_tool_call(0, 0, "id", "update_spec", {"patch": {"question": "q"}}, "ok", True)
    # Successful lookups are suppressed entirely — no print, no spinner pause.
    obs.on_tool_call(0, 1, "id", "list_metrics", {}, "…20 metrics…", True)
    out = console.file.getvalue()
    assert "updated plan" in out and "✓" in out and "question" in out
    assert "list_metrics" not in out
    # spinner paused and resumed only around the one rendered line
    assert status.events.count("stop") == 1 and status.events.count("start") == 1


def test_observer_collapses_recovered_failures() -> None:
    # A transient failure followed by a success is dropped silently (the
    # agent self-corrected); only a "self-correcting" spinner note appears.
    console = _console()
    status = _FakeStatus()
    obs = repl.ReplObserver(console, status)
    obs.on_tool_call(0, 0, "id", "finalize_spec", {}, "ERROR: structural errors", False)
    assert any("self-correcting (1)" in e for e in status.events)
    obs.on_tool_call(0, 1, "id", "remove_spec_fields", {"keys": ["notes"]}, "ok", True)
    out = console.file.getvalue()
    assert "finalize" not in out and "✗" not in out  # failure never surfaced
    assert "removed from plan" in out  # the recovery did


def test_observer_surfaces_failure_unrecovered_at_turn_end() -> None:
    console = _console()
    status = _FakeStatus()
    obs = repl.ReplObserver(console, status)
    obs.on_tool_call(0, 0, "id", "bash", {"command": "ls"}, "ERROR: boom", False)
    assert "boom" not in console.file.getvalue()  # held back, not yet shown
    obs.on_final(0, "done")  # turn ends without recovery → surface it
    out = console.file.getvalue()
    assert "✗" in out and "boom" in out and "run bash" in out


def test_observer_flushes_after_threshold_consecutive_failures() -> None:
    console = _console()
    status = _FakeStatus()
    obs = repl.ReplObserver(console, status)
    for i in range(repl._FAILURE_FLUSH_THRESHOLD):
        obs.on_tool_call(0, i, "id", "update_spec", {}, f"ERROR: bad {i}", False)
    out = console.file.getvalue()
    # Distinct failures are surfaced (no recovery), not hidden forever.
    assert out.count("✗") == repl._FAILURE_FLUSH_THRESHOLD


def test_observer_collapses_identical_failure_flood() -> None:
    # A stuck agent repeating the SAME malformed call reads as one line ×N,
    # not a wall — but is still honestly surfaced.
    console = _console()
    status = _FakeStatus()
    obs = repl.ReplObserver(console, status)
    err = "ERROR: compute_metric: split (non-empty string) is required"
    for i in range(repl._FAILURE_FLUSH_THRESHOLD):
        obs.on_tool_call(0, i, "id", "compute_metric", {"metric": "logit_diff"}, err, False)
    out = console.file.getvalue()
    assert out.count("✗") == 1  # collapsed
    assert f"×{repl._FAILURE_FLUSH_THRESHOLD}" in out  # with the count


def test_highlight_identifiers_wraps_snake_case_outside_code() -> None:
    src = (
        "Use logit_diff and feature_activation_density now. "
        "Keep `already_code` and a path /tmp/x.\n"
        "```\nraw_block_id stays raw\n```\n"
        "Also activation_cache here."
    )
    out = repl._highlight_identifiers(src)
    assert "`logit_diff`" in out
    assert "`feature_activation_density`" in out
    assert "`activation_cache`" in out
    assert out.count("`already_code`") == 1  # not double-wrapped
    assert "`raw_block_id`" not in out  # inside the fence, untouched
    # plain words and non-identifier text are left alone
    assert "Use" in out and "now" in out


def test_highlight_leaves_markdown_emphasis_alone() -> None:
    # Single-underscore emphasis must not be mangled into code.
    assert repl._highlight_identifiers("_italic_ text") == "_italic_ text"
    # A word with no underscore is not an identifier.
    assert repl._highlight_identifiers("just steering here") == "just steering here"


def test_render_answer_highlights_identifiers_in_cyan() -> None:
    import io as _io

    from rich.console import Console as _C

    console = _C(file=_io.StringIO(), force_terminal=True, width=80)
    repl.render_answer(console, "Compute logit_diff for the contrast.")
    out = console.file.getvalue()
    assert "logit_diff" in out
    assert "\x1b[36m" in out or ";36" in out  # cyan ansi was emitted


def test_quiet_loop_handler_drops_benign_noise_keeps_real_errors() -> None:
    """A GC'd background task (no 'exception' in the context) must be
    swallowed so it can't trigger prompt_toolkit's 'Press ENTER' UI freeze;
    genuine exceptions still reach the previous handler."""

    async def go() -> None:
        loop = asyncio.get_running_loop()
        delegated: list[dict] = []
        loop.set_exception_handler(lambda lp, ctx: delegated.append(ctx))
        repl.install_quiet_loop_handler()
        handler = loop.get_exception_handler()
        # Benign events — dropped, never delegated.
        handler(loop, {"message": "Task was destroyed but it is pending!"})
        handler(loop, {"message": "x", "exception": None})
        assert delegated == []
        # A real exception — delegated to the previous handler.
        err = ValueError("boom")
        handler(loop, {"message": "real", "exception": err})
        assert len(delegated) == 1 and delegated[0]["exception"] is err

    asyncio.run(go())


def test_format_tool_event_semantics() -> None:
    fmt = repl._format_tool_event
    # Lookups inform the agent, not the user → silent on success…
    for tool in ("list_metrics", "show_spec", "describe_spec", "read_file",
                 "validate_spec", "read_skill"):
        assert fmt(tool, {}, True) is None, tool
    # …but never silent on failure, and the reason is shown — in plain
    # language on both sides of the dash (the agent still gets the precise
    # original error; only the feed is humanized).
    line = fmt("validate_spec", {}, False, "ERROR: contrast: Field required\ntrace")
    assert line and "✗" in line and "contrast: Field required" in line
    assert "couldn't validate the plan" in line
    line = fmt(
        "update_spec", {}, False,
        "ERROR: patch contains keys that are not InvestigationSpec fields: 'x'",
    )
    assert "couldn't update the plan" in line
    assert "not plan fields" in line and "InvestigationSpec" not in line
    # State changes render as verb phrases; field names are humanized
    # (no underscores) and "spec" reads as "plan" per the user convention.
    line = fmt("update_spec", {"patch": {"success_criteria": [], "contrast": {}}}, True)
    assert "updated plan" in line and "success criteria, contrast" in line
    assert "success_criteria" not in line
    assert "removed from plan" in fmt("remove_spec_fields", {"keys": ["budget"]}, True)
    # finalize_spec is single-phase now: a successful call always means the
    # plan was written, so one approval line covers it.
    assert "approved and locked in" in fmt("finalize_spec", {}, True)
    assert "bash[/bold][dim] · ls -la" in fmt("bash", {"command": "ls -la"}, True)
    # A multi-line script reads as distinct commands, not a flattened run-on.
    multi = fmt("bash", {"command": "ls -la\nls -la prompt_batches || true\nwc -l x"}, True)
    assert "ls -la ; ls -la prompt_batches || true ; wc -l x" in multi
    assert "ls -la ls -la" not in multi
    assert "scripts/x.py" in fmt("write_file", {"path": "scripts/x.py"}, True)
    # Investigation Tier-2 tools — now a primary surface for the feed (the
    # auto-launched run renders through this).
    assert "computed accuracy" in fmt("compute_metric", {"metric": "accuracy"}, True)
    assert "computed accuracy" in fmt(
        "compute_and_commit_metric", {"metric": "accuracy"}, True
    )
    # The artifact kind is humanized in the feed too (no raw CamelCase).
    assert "committed metric result" in fmt(
        "commit_artifact", {"kind": "MetricResult"}, True
    )
    assert "committed intervention result" in fmt(
        "commit_artifact", {"kind": "InterventionResult"}, True
    )
    assert "evaluated criterion" in fmt(
        "evaluate_criterion", {"criterion_id": "c1"}, True
    )
    assert "advanced to next stage" in fmt("advance_stage", {}, True)
    assert "requested spec revision" in fmt("request_spec_revision", {"reason": "x"}, True)
    assert fmt("current_stage", {}, True) is None  # read-only view stays silent
    assert "couldn't compute the metric" in fmt(
        "compute_metric", {}, False, "ERROR: logit_diff: missing required input keys"
    )
    # Unknown (e.g. MCP) tools keep the generic rendering rather than hiding.
    assert "mcp_thing" in fmt("mcp_thing", {"x": 1}, True)


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


def test_fanout_observer_dispatches_and_isolates_failures() -> None:
    seen: list[str] = []

    class A:
        def on_tool_call(self, *a) -> None:
            seen.append("A.tool")

        def on_final(self, *a) -> None:
            raise RuntimeError("boom")  # one bad observer must not break others

    class B:
        def on_tool_call(self, *a) -> None:
            seen.append("B.tool")

    fan = repl._FanoutObserver([A(), B()])
    fan.on_tool_call(0, 0, "id", "bash", {}, "out", True)
    assert seen == ["A.tool", "B.tool"]
    fan.on_final(0, "text")  # A raises, swallowed; B lacks on_final — no error


def test_run_live_turn_fans_out_and_returns_answer(monkeypatch) -> None:
    disk_events: list[str] = []

    class Disk:
        def on_tool_call(self, *a) -> None:
            disk_events.append("tool")

        def on_final(self, *a) -> None:
            disk_events.append("final")

    async def fake_turn(prompt, config, context, router, observer=None, cost_tracker=None):
        observer.on_tool_call(0, 0, "id", "update_spec", {"patch": {"q": 1}}, "ok", True)
        observer.on_final(0, "done")
        return "done"

    monkeypatch.setattr(repl, "run_agent_turn", fake_turn)
    console = _console()
    ctx = ContextManager(skill_registry=SkillRegistry({}))
    out = asyncio.run(
        repl.run_live_turn(
            "p", AgentConfig(), ctx, object(), console, extra_observer=Disk()
        )
    )
    assert out == "done"
    # The transcript observer saw the events…
    assert disk_events == ["tool", "final"]
    # …and the live feed rendered the semantic line.
    assert "updated plan" in console.file.getvalue()


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


# ---------------------------------------------------------------------------
# /temperature
# ---------------------------------------------------------------------------


def test_temperature_set_show_reset() -> None:
    console = _console()
    ui = _ui(console, config=AgentConfig(model_name="anthropic/claude-sonnet-4-5"))
    assert ui.config.temperature is None
    asyncio.run(repl.handle_repl_command("/temperature 0.4", ui))
    assert ui.config.temperature == 0.4
    asyncio.run(repl.handle_repl_command("/temperature", ui))  # show
    assert "0.4" in console.file.getvalue()
    asyncio.run(repl.handle_repl_command("/temperature default", ui))  # reset
    assert ui.config.temperature is None


def test_temperature_rejects_out_of_range_and_garbage() -> None:
    console = _console()
    ui = _ui(console, config=AgentConfig(model_name="anthropic/claude-sonnet-4-5",
                                         temperature=0.5))
    asyncio.run(repl.handle_repl_command("/temperature 5", ui))
    assert ui.config.temperature == 0.5  # unchanged
    asyncio.run(repl.handle_repl_command("/temperature hot", ui))
    assert ui.config.temperature == 0.5  # unchanged
    out = console.file.getvalue()
    assert "between 0.0 and 2.0" in out and "Usage" in out


def test_temperature_warns_for_reasoning_model() -> None:
    console = _console()
    ui = _ui(console, config=AgentConfig(model_name="openai/gpt-5-nano"))
    asyncio.run(repl.handle_repl_command("/temperature 0.4", ui))
    out = console.file.getvalue()
    assert ui.config.temperature == 0.4  # stored (applies if they switch models)
    assert "reasoning model" in out  # but warned it's ignored here


def test_temperature_status_helper() -> None:
    assert "provider default" in repl._temperature_status(AgentConfig())
    assert "0.5" in repl._temperature_status(AgentConfig(temperature=0.5))
    ignored = repl._temperature_status(
        AgentConfig(model_name="openai/gpt-5-nano", temperature=0.5)
    )
    assert "ignored" in ignored and "reasoning" in ignored


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
# word wrap (input box)
# ---------------------------------------------------------------------------


def test_word_wrap_consumes_breaking_space() -> None:
    # "…what do| they" — the inter-word space must not lead the next row.
    rows, _ = repl._word_wrap("how many total phases do we have", width=20)
    assert rows == ["how many total ", "phases do we have"]
    for row in rows[1:]:
        assert not row.startswith(" ")


def test_word_wrap_keeps_words_whole() -> None:
    rows, _ = repl._word_wrap("alpha beta gamma delta", width=11)
    # Final row is exactly full → a fresh empty row hosts the end cursor.
    assert rows == ["alpha beta ", "gamma delta", ""]


def test_word_wrap_hard_splits_overlong_words() -> None:
    rows, _ = repl._word_wrap("supercalifragilistic", width=8)
    assert rows == ["supercal", "ifragili", "stic"]


def test_word_wrap_cursor_map_is_consistent() -> None:
    text = "one two three four"
    width = 8
    rows, pos = repl._word_wrap(text, width)
    assert len(pos) == len(text) + 1
    for row, col in pos:
        assert 0 <= row < len(rows) and 0 <= col <= width
    # End-of-text cursor lands at the end of the last row.
    assert pos[len(text)] == (len(rows) - 1, len(rows[-1]))
    # Non-space characters map onto the cell that actually renders them.
    for i, ch in enumerate(text):
        row, col = pos[i]
        if ch != " " and col < len(rows[row]):
            assert rows[row][col] == ch


def test_word_wrap_exact_fit_moves_cursor_to_fresh_row() -> None:
    rows, pos = repl._word_wrap("abcdefgh", width=8)
    assert rows == ["abcdefgh", ""]
    assert pos[8] == (1, 0)  # end cursor on a real cell, not past the edge


def test_input_box_screen_cursor_tracks_typing(tmp_path: Path) -> None:
    """Regression: the rendered cursor must sit at the typed position.

    The custom word-wrap control once emitted cursor cells that were never
    written to the screen, so prompt_toolkit's lookup missed and parked the
    visible cursor at the window origin (top-left) forever.
    """
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout.containers import WritePosition
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen
    from prompt_toolkit.output import DummyOutput

    width, prefix = 40, 4  # window cols; "│ ❯ " prefix cells

    def cursor_for(text: str):
        from prompt_toolkit.application.current import set_app

        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                box = repl.BoxedInput(_ui(), tmp_path / "history")
                box.buffer.text = text
                box.buffer.cursor_position = len(text)
                screen = Screen()
                # The cursor only registers for the focused window of the
                # active app — enter the box's own Application for the render.
                with set_app(box.app):
                    box._input_window.write_to_screen(
                        screen, MouseHandlers(),
                        WritePosition(xpos=0, ypos=0, width=width, height=8),
                        "", False, None,
                    )
                return screen.cursor_positions[box._input_window]

    empty = cursor_for("")
    assert (empty.x, empty.y) == (prefix, 0)

    typed = cursor_for("hello")
    assert (typed.x, typed.y) == (prefix + 5, 0)  # NOT stuck at the origin

    long_text = "What is the difference between Phase 1 and Phase 2 anyway"
    rows, pos = repl._word_wrap(long_text, width - prefix)
    expect_row, expect_col = pos[len(long_text)]
    wrapped = cursor_for(long_text)
    assert (wrapped.x, wrapped.y) == (prefix + expect_col, expect_row)
    assert wrapped.y > 0  # genuinely on a wrapped row


def test_word_wrap_empty_and_newlines() -> None:
    rows, pos = repl._word_wrap("", width=10)
    assert rows == [""] and pos == [(0, 0)]
    rows, _ = repl._word_wrap("a\nb", width=10)
    assert rows == ["a b"]  # pasted newlines display as spaces


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


def test_investigation_crash_does_not_kill_session(tmp_path: Path, monkeypatch) -> None:
    """An investigation that raises must NOT take the session down — it's caught,
    surfaced, and the REPL returns to the prompt. (The run-1 NaN crash propagated
    all the way out of run_investigation and WOULD have killed the session.)"""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    console = _console()
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    registry = SkillRegistry({})
    context = ContextManager(skill_registry=registry, model_name=config.model_name)
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()

    async def fake_turn(prompt, config, context, router, observer=None, **kwargs):
        (spec_dir / "demo_rev1.json").write_text("{}")  # finalize → on_finalize fires
        return "locked in"

    monkeypatch.setattr(repl, "run_agent_turn", fake_turn)

    crashed = {"called": False}

    async def boom(spec_path):
        crashed["called"] = True
        raise RuntimeError("NaN exploded the run")

    async def go():
        with create_pipe_input() as pipe:
            pipe.send_text("approve\r/exit\r")  # launch (crashes), then leave
            with create_app_session(input=pipe, output=DummyOutput()):
                return await repl.run_spec_repl(
                    config=config, context=context, router=object(), console=console,
                    registry=registry, spec_dir=spec_dir, max_turns=5,
                    state_dir=tmp_path, on_finalize=boom,
                )

    result = asyncio.run(go())
    out = console.file.getvalue()
    assert crashed["called"]  # the investigation DID run and raised
    assert result is None  # the session exited cleanly via /exit — NOT a crash
    assert "stopped on an error" in out and "still alive" in out
    assert "NaN exploded" in out  # the real reason surfaced
    assert "Back to design" in out  # returned to the prompt afterwards


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


def test_repl_resume_replays_full_history(tmp_path: Path) -> None:
    """--continue must show the whole prior conversation (Claude-Code
    style), not just restore it invisibly into the agent's context."""
    payload = {
        "session_id": "s",
        "title": "surprise",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:10:00Z",
        "model_name": "m",
        "turn": 2,
        "messages": [
            {"role": "user", "content": "How do models represent surprise?"},
            {
                "role": "assistant",
                "content": "Let me check the schema first.",
                "tool_calls": [
                    {"id": "t1", "function": {
                        "name": "update_spec",
                        "arguments": '{"patch": {"question": "q"}}',
                    }},
                    {"id": "t2", "function": {
                        "name": "validate_spec", "arguments": "{}",
                    }},
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "name": "update_spec",
             "content": "ok"},
            {"role": "tool", "tool_call_id": "t2", "name": "validate_spec",
             "content": "ERROR: contrast is required"},
            {"role": "assistant", "content": "Here is the **draft plan**."},
            {"role": "user", "content": "looks good, refine the dataset"},
        ],
        "cost": {"total_cost_usd": 0.0, "total_requests": 0, "by_model": {}},
        "draft": None,
    }
    result, output, _ = _run_repl_with_keys(
        "/exit\r", tmp_path, resume_payload=payload,
    )
    assert result is None
    assert "previous conversation" in output
    # Both user prompts replayed in the input-box style.
    assert "How do models represent surprise?" in output
    assert "looks good, refine the dataset" in output
    # Interim thought, semantic tool feed with outcomes, markdown answer.
    assert "Let me check the schema first." in output
    assert "updated plan" in output and "✓" in output
    # The failed call shows what couldn't happen and why, in plain language.
    assert "couldn't validate the plan" in output and "✗" in output
    assert "contrast is required" in output
    assert "draft plan" in output
    assert "session restored" in output


def test_repl_on_finalize_runs_inline_and_stays_in_session(tmp_path, monkeypatch) -> None:
    """A finalized spec runs the investigation in-session (on_finalize) and the
    loop CONTINUES — the user is never dumped to the shell."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    console = _console()
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    registry = SkillRegistry({})
    context = ContextManager(skill_registry=registry, model_name=config.model_name)
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()

    turn_calls = {"n": 0}

    async def fake_turn(prompt, config, context, router, observer=None, **kwargs):
        turn_calls["n"] += 1
        if turn_calls["n"] == 1:
            (spec_dir / "demo_rev1.json").write_text("{}")  # finalize
            return "spec approved"
        return "a follow-up answer"

    monkeypatch.setattr(repl, "run_agent_turn", fake_turn)

    finalized: list[str] = []

    async def on_finalize(spec_path: str) -> None:
        finalized.append(spec_path)

    async def go():
        with create_pipe_input() as pipe:
            pipe.send_text("design it\rtell me more\r/exit\r")
            with create_app_session(input=pipe, output=DummyOutput()):
                return await repl.run_spec_repl(
                    config=config, context=context, router=object(), console=console,
                    registry=registry, spec_dir=spec_dir, max_turns=5,
                    state_dir=tmp_path, on_finalize=on_finalize,
                )

    result = asyncio.run(go())
    assert result is None  # stayed in session, did not return the spec path
    assert finalized == [str(spec_dir / "demo_rev1.json")]  # investigation ran inline
    assert turn_calls["n"] == 2  # a follow-up turn happened AFTER the investigation
    assert "Back to design" in console.file.getvalue()


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


def test_looks_like_approval() -> None:
    for yes in ("Approve.", "approve", "yes", "yes go ahead", "go ahead",
                "lock it in", "sounds good", "Approved!", "ok", "lgtm",
                "go ahead, revise it and re-run",
                # natural go-aheads that must also count (the "Continue" bug)
                "Continue.", "continue", "keep going", "go", "let's go",
                "proceed", "run it", "launch it", "finalize", "ship it",
                "go for it", "lock it",
                # an explicit approval phrase inside a longer reply (the user
                # answered the agent's questions AND approved in one message)
                "I want a local model. Around 300 is fine. No domain focus. I approve otherwise.",
                "Looks good, go ahead and run it on gpt2.",
                "that all sounds good to me, lgtm"):
        assert repl._looks_like_approval(yes), yes
    for no in ("yes, but use a bigger model", "approve after you change the model",
               "how does this work?", "use distilgpt2 instead", "", "no",
               "wait, change the dataset", "can you explain the metric first?",
               "continue but switch the model first",
               # an approval with a caveat is NOT a clean go-ahead
               "I approve, but change the model to gpt2-large first",
               "I'll approve once you fix the dataset"):
        assert not repl._looks_like_approval(no), no


def test_resume_hint_shows_and_persists_on_exit(tmp_path) -> None:
    """On exit the user gets the exact command to resume THIS session — and the
    session is force-persisted so `--continue <id>` resolves EVEN at turn 0 (the
    'I Ctrl-C'd immediately and saw nothing' bug)."""
    import types

    from autointerp_agent.sessions import SessionStore

    store = SessionStore(root=tmp_path)
    ui = types.SimpleNamespace(
        session_store=store, session_id="sess-1", turn=0,  # exited before a turn
        context=types.SimpleNamespace(messages=[]),
        config=types.SimpleNamespace(model_name="gpt2"),
        cost=types.SimpleNamespace(to_dict=lambda: {}),
        spec_dir=None,
    )
    console = _console()
    repl._print_resume_hint(console, ui)
    assert "autointerp --continue sess-1" in console.file.getvalue()
    assert store.load("sess-1") is not None  # persisted → the id resolves

    # No hint without a session store (non-interactive caller).
    c3 = _console()
    repl._print_resume_hint(
        c3, types.SimpleNamespace(session_store=None, session_id="x", turn=5)
    )
    assert c3.file.getvalue() == ""


def test_resume_hint_renders_uniform_color(tmp_path) -> None:
    """The id must not be two-toned: rich's number highlighter used to tint only
    the digit runs (20260615/153036) cyan, leaving the dashes + hex 'f71f' white.
    The whole command should render as one uniform span."""
    import io
    import types

    from rich.console import Console

    from autointerp_agent.sessions import SessionStore

    sid = "20260615-153036-f71f"
    ui = types.SimpleNamespace(
        session_store=SessionStore(root=tmp_path), session_id=sid, turn=0,
        context=types.SimpleNamespace(messages=[]),
        config=types.SimpleNamespace(model_name="gpt2"),
        cost=types.SimpleNamespace(to_dict=lambda: {}),
        spec_dir=None,
    )
    console = Console(file=io.StringIO(), force_terminal=True, width=120,
                      color_system="standard")
    repl._print_resume_hint(console, ui)
    raw = console.file.getvalue()
    # the command (incl. the id) is one contiguous styled run — no ANSI breaks
    # splitting the digits from the dashes/hex the way the highlighter did.
    assert f"autointerp --continue {sid}" in raw
    assert "\x1b[1m-" not in raw  # the old bold-default dash artifact is gone


class _FakeRouter:
    """Minimal router stub: records call_tool calls and returns a fixed result,
    optionally running a side effect (e.g. writing a spec file) first."""

    def __init__(self, result, side_effect=None):
        self.result = result
        self.side_effect = side_effect
        self.calls: list = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if self.side_effect is not None:
            self.side_effect()
        return self.result


def test_force_finalize_locks_in_on_user_approval(tmp_path) -> None:
    """When the user approved but the agent didn't finalize, the REPL locks the
    plan in deterministically via finalize_spec — no reliance on the model."""
    import asyncio

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    before = repl._snapshot_specs(spec_dir)

    def _write_spec() -> None:
        (spec_dir / "plan_rev1.json").write_text("{}")

    router = _FakeRouter(("Plan finalized as revision 1, saved to …", True), _write_spec)
    new = asyncio.run(repl._force_finalize(router, _console(), spec_dir, before))
    assert any(p.name == "plan_rev1.json" for p in new)  # the new spec is returned
    assert router.calls[0][0] == "finalize_spec"
    assert router.calls[0][1]["approver_kind"] == "human"  # the human approved


def test_force_finalize_noop_when_draft_not_ready(tmp_path) -> None:
    """If the draft isn't finalize-ready, force-finalize writes nothing and shows
    a PLAIN-LANGUAGE note (not a raw pydantic 'structural errors' blob) — a
    half-formed plan is never launched on a stray 'continue'."""
    import asyncio

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    before = repl._snapshot_specs(spec_dir)
    router = _FakeRouter(("Cannot finalize — spec has structural errors:\nfoo", False))
    console = _console()
    new = asyncio.run(repl._force_finalize(router, console, spec_dir, before))
    assert new == set()
    out = console.file.getvalue()
    assert "isn't fully filled in yet" in out  # plain language
    assert "structural error" not in out.lower()  # no raw blob leaked

    # a NON-structural blocker is shown (cleaned), so real issues aren't hidden
    c2 = _console()
    r2 = _FakeRouter(("Cannot finalize — metric x is not runnable", False))
    asyncio.run(repl._force_finalize(r2, c2, spec_dir, before))
    assert "metric x is not runnable" in c2.file.getvalue()


def test_humanize_identifiers_translates_leaked_schema_names() -> None:
    """Internal CamelCase identifiers leaked into PROSE become plain language —
    including the hallucinated 'InvestigationPlan' that no detect-list covers.
    Distinctive terms are COLORED (backtick-wrapped → accent inline code) like
    the underscored identifiers; everyday single words stay plain."""
    h = repl._humanize_identifiers
    # distinctive schema/type terms: humanized AND colored
    assert h("draft a concrete InvestigationPlan") == "draft a concrete `investigation plan`"
    assert h("commit an InterventionResult") == "commit an `intervention result`"
    assert h("update the InvestigationSpec") == "update the `investigation plan`"
    assert h("the MetricResult and CandidateSite") == "the `metric result` and `candidate site`"
    # EVERY casing/spacing of "investigation plan" normalizes to the colored
    # lowercase term — incl. the spaced Title Case form a real run leaked
    # ("convert this into a formal Investigation Plan").
    assert h("a formal Investigation Plan") == "a formal `investigation plan`"
    assert h("the investigation plan here") == "the `investigation plan` here"
    # no double-wrap when several forms appear together
    assert "``" not in h("InvestigationSpec, Investigation Plan, InvestigationPlan")
    # everyday single words: humanized but NOT colored
    assert h("ModelRef on DatasetSpec") == "model on dataset"
    assert "investigation plan" in h("I'll build the InvestigationPlan now.")


def test_humanize_identifiers_preserves_code_and_plain_english() -> None:
    """Genuine `code` references and ordinary words must be left alone."""
    h = repl._humanize_identifiers
    # inline code + fenced code are untouched (the agent may show real schema code)
    assert h("see `InvestigationSpec` here") == "see `InvestigationSpec` here"
    assert "InvestigationSpec" in h("```\nspec = InvestigationSpec()\n```")
    # the ordinary English word, planning prose, and model names never match
    assert h("we ran 3 Investigations") == "we ran 3 Investigations"
    assert h("the investigation planning process") == "the investigation planning process"
    assert h("using GPT2 / gpt-neo-125M") == "using GPT2 / gpt-neo-125M"


def test_render_answer_shows_plain_language_not_identifiers() -> None:
    """End to end: the user never sees 'InvestigationPlan' in a rendered reply."""
    console = _console(width=120)
    repl.render_answer(console, "I'll draft a concrete InvestigationPlan with stages.")
    out = console.file.getvalue()
    assert "investigation plan" in out
    assert "InvestigationPlan" not in out


def test_write_file_event_shows_run_dir_location() -> None:
    """During an investigation, a write to `scripts/x.py` is shown under the run
    dir so it can't be mistaken for the repo's own scripts/."""
    line = repl._format_tool_event(
        "write_file", {"path": "scripts/step0_setup.py"}, True,
        run_dir="runs/my_run_rev1",
    )
    assert "runs/my_run_rev1/scripts/step0_setup.py" in line
    # No run dir (plain REPL) → raw path, unchanged.
    line2 = repl._format_tool_event("write_file", {"path": "scripts/x.py"}, True)
    assert "scripts/x.py" in line2 and "runs/" not in line2
    # Absolute paths are left as-is.
    line3 = repl._format_tool_event(
        "write_file", {"path": "/tmp/abs.py"}, True, run_dir="runs/r1"
    )
    assert "/tmp/abs.py" in line3
