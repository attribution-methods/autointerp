"""Tests for the investigation driver: drive-to-completion loop, autonomy
prompt, and the run header/summary panels. No LLM — turns are faked."""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from autointerp.pipelines.investigation import init_run
from autointerp.pipelines.investigation.main import build_system_prompt
from autointerp.pipelines.investigation.state import (
    TerminalState,
    read_state,
    write_state,
)
from autointerp.schemas import BehaviorSpec
from autointerp.spec import (
    Approval,
    Budget,
    ContrastSpec,
    Criterion,
    DatasetSpec,
    InvestigationSpec,
    InvestigationStage,
    MetricName,
    ModelRef,
    PatternId,
    SpecStatus,
    StageSpec,
    ToolName,
)
from autointerp_agent import investigation as inv


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=100)


def _spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="driver-fixture",
        revision=1,
        question="How does the model represent X?",
        hypothesis="h",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="gpt2"),
        dataset=DatasetSpec(dataset_id="ds", source="generated", n_samples=10),
        contrast=ContrastSpec(
            contrast_id="c", positive_template="{x}", negative_template="{y}"
        ),
        stages=[
            StageSpec(
                stage=InvestigationStage.BLACK_BOX,
                pattern=PatternId.BLACKBOX_THEN_PATCHING,
                tools=[ToolName.BLACKBOX_PROBE],
                metrics=[MetricName.ACCURACY],
            )
        ],
        success_criteria=[
            Criterion(
                criterion_id="c1", description="d", metric=MetricName.ACCURACY,
                comparator=">=", threshold=0.5, on_split="dev",
            )
        ],
        budget=Budget(max_tool_calls=50),
        status=SpecStatus.APPROVED,
        approval=Approval(
            approver="t", approver_kind="agent",
            approved_at=datetime.now(timezone.utc),
        ),
    )


# ---------------------------------------------------------------------------
# drive-to-completion loop
# ---------------------------------------------------------------------------


def test_drive_stops_at_terminal_and_nudges(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path / "runs")
    prompts: list[str] = []

    async def fake_turn(prompt: str) -> str:
        prompts.append(prompt)
        if len(prompts) == 2:  # reach a terminal state on the second turn
            st = read_state(handle.state_path)
            st.terminal_state = TerminalState.COMPLETED
            write_state(handle.state_path, st)
        return "ok"

    answer, reason = asyncio.run(
        inv._drive_to_completion(handle, fake_turn, "begin", max_continuations=5)
    )
    assert reason == "terminal"
    assert prompts == ["begin", inv._CONTINUE_NUDGE]  # nudged once, then terminal


def test_drive_respects_continuation_cap(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path / "runs")
    n = 0

    async def fake_turn(prompt: str) -> str:
        nonlocal n
        n += 1
        return "still chatting, never reaching terminal"

    answer, reason = asyncio.run(
        inv._drive_to_completion(handle, fake_turn, "begin", max_continuations=3)
    )
    assert reason == "continuation_cap"
    assert n == 4  # initial + 3 continuations, then give up


def test_drive_stops_on_inner_max_iterations(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path / "runs")

    async def fake_turn(prompt: str) -> str:
        return "Stopped after max_iterations without a final answer."

    answer, reason = asyncio.run(
        inv._drive_to_completion(handle, fake_turn, "begin")
    )
    assert reason == "max_iterations"


# ---------------------------------------------------------------------------
# autonomy instruction in the system prompt
# ---------------------------------------------------------------------------


def test_system_prompt_demands_autonomy() -> None:
    prompt = build_system_prompt(_spec(), "run-x")
    low = prompt.lower()
    assert "autonomous" in low
    assert "never ask the user" in low
    # The pre-existing inviolable rules and spec are still present.
    assert "Inviolable rules" in prompt
    assert "request_spec_revision" in prompt


def test_system_prompt_steers_metrics_to_one_shot_tool() -> None:
    """Agents waste many turns hand-building MetricResult payloads. The prompt
    must steer them to the preferred one-shot tool and state that `split` is a
    top-level argument, not a payload field."""
    prompt = build_system_prompt(_spec(), "run-x")
    assert "compute_and_commit_metric" in prompt
    assert "top-level argument" in prompt


# ---------------------------------------------------------------------------
# run header / summary panels
# ---------------------------------------------------------------------------


def test_run_header_panel_renders(tmp_path: Path) -> None:
    spec = _spec()
    handle = init_run(spec, runs_root=tmp_path / "runs")
    console = _console()
    inv._print_run_header(console, spec, handle)
    out = console.file.getvalue()
    assert "running investigation" in out
    assert "driver-fixture" in out
    assert "black_box" in out


def test_run_summary_terminal_and_paused(tmp_path: Path) -> None:
    spec = _spec()
    handle = init_run(spec, runs_root=tmp_path / "runs")

    state = read_state(handle.state_path)
    state.terminal_state = TerminalState.COMPLETED
    console = _console()
    inv._print_run_summary(
        console, handle, state, "terminal", tmp_path / "report.json",
        spec_arg="outputs/specs/s.json",
    )
    out = console.file.getvalue()
    assert "investigation finished" in out
    assert "completed" in out
    assert f"runs show {handle.run_id}" in out

    paused_console = _console()
    inv._print_run_summary(
        paused_console, handle, read_state(handle.state_path),
        "continuation_cap", None, spec_arg="outputs/specs/s.json",
    )
    paused = paused_console.file.getvalue()
    assert "paused" in paused
    assert "investigate --spec outputs/specs/s.json" in paused


def test_criteria_tally_handles_passed_bool(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path / "runs")
    state = read_state(handle.state_path)
    assert inv._criteria_tally(state) == (0, 0)  # nothing evaluated yet


# ---------------------------------------------------------------------------
# end-to-end async_main (real gates + driver + panels; fake LLM, no GPU)
# ---------------------------------------------------------------------------


def test_render_inline_results_shows_criteria_and_findings(tmp_path: Path) -> None:
    import json

    handle = init_run(_spec(), runs_root=tmp_path / "runs")
    # A report.json as write_report would produce.
    (handle.root / "report.json").write_text(json.dumps({
        "claims": ["L9H9 is the dominant name-mover head."],
        "metadata": {"criteria_evaluated": {
            "c1": {"metric": "accuracy", "comparator": ">=", "threshold": 0.5,
                   "value": 0.98, "verdict": "pass"},
        }},
    }))
    state = read_state(handle.state_path)
    console = _console()
    inv._render_inline_results(console, handle, state)
    out = console.file.getvalue()
    assert "success criteria" in out.lower()
    assert "accuracy" in out and "0.98" in out and "pass" in out
    assert "Findings" in out and "name-mover" in out


def test_async_main_drives_real_gates_to_completion(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The whole auto-launch path: a fake 'agent' drives the real Tier-2
    gates to a terminal state, and async_main writes the report and renders
    the finished panel — no LLM, no model load."""
    import json

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec().model_dump_json(indent=2))
    runs_root = tmp_path / "runs"

    async def fake_ready(config, console, *, interactive):
        return config

    async def fake_turn(prompt, config, context, router, observer=None, cost_tracker=None):
        # Reach `completed` through the registered gates, as a real agent would.
        out, ok = await router.call_tool(
            "compute_metric",
            {"metric": "accuracy", "metric_id": "a1",
             "inputs": {"predictions": [1, 1], "labels": [1, 1]}},
        )
        assert ok, out
        parsed = json.loads(out)
        await router.call_tool(
            "commit_artifact",
            {"kind": "MetricResult", "payload": parsed["payload"],
             "split": "dev", "provenance_token": parsed["provenance_token"]},
        )
        await router.call_tool("evaluate_criterion", {"criterion_id": "c1"})
        _, adv_ok = await router.call_tool("advance_stage", {})
        assert adv_ok
        if observer is not None:
            observer.on_tool_call(0, 0, "id", "advance_stage", {}, "ok", True)
            observer.on_final(0, "Investigation complete.")
        return "Investigation complete — accuracy 1.0 on dev."

    monkeypatch.setattr(inv, "ensure_model_ready", fake_ready)
    monkeypatch.setattr(inv, "run_agent_turn", fake_turn)
    monkeypatch.setattr("autointerp_agent.repl.run_agent_turn", fake_turn)

    rc = inv.main([
        "--spec", str(spec_path), "--runs-root", str(runs_root), "--auto-approve",
    ])
    assert rc == 0
    # Report written for a terminal run.
    report = runs_root / "driver-fixture_rev1" / "report.json"
    assert report.exists()
    # The finished panel rendered with the terminal state.
    out = capsys.readouterr().out
    assert "running investigation" in out  # start header
    assert "investigation finished" in out and "completed" in out
