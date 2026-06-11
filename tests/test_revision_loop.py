"""Tests for the auto-revision-loop helpers.

Only the pure-data helpers are tested here. The actual Stage-0 re-engagement
loop hits an LLM and lives behind ``cmd_investigate``; that path is covered
indirectly by the existing entrypoint tests + tested manually.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from autointerp.pipelines.investigation import init_run
from autointerp.pipelines.investigation.metrics import compute_and_commit_metric
from autointerp.pipelines.investigation.revision import request_spec_revision
from autointerp.pipelines.investigation.state import read_state
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
from autointerp_agent.revision_loop import (
    REVISABLE_TERMINAL_STATES,
    should_auto_revise,
    summarize_for_stage0,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec(spec_id: str = "rev-loop-fixture") -> InvestigationSpec:
    return InvestigationSpec(
        spec_id=spec_id,
        revision=1,
        question="Does the model X?",
        hypothesis="L9H9 mediates X.",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="m"),
        dataset=DatasetSpec(
            dataset_id="ds", source="generated", n_samples=10, split="dev", seed=1
        ),
        contrast=ContrastSpec(
            contrast_id="c",
            positive_template="{x}",
            negative_template="{y}",
            pairing="matched",
        ),
        stages=[
            StageSpec(
                stage=InvestigationStage.BLACK_BOX,
                pattern=PatternId.BLACKBOX_THEN_PATCHING,
                tools=[ToolName.BLACKBOX_PROBE],
                metrics=[MetricName.ACCURACY],
            ),
        ],
        success_criteria=[
            Criterion(
                criterion_id="c1",
                description="dev acc",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.9,
                on_split="dev",
            ),
        ],
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=_now()),
    )


def test_should_auto_revise_terminal_states() -> None:
    assert should_auto_revise(None) is False
    assert should_auto_revise({"terminal_state": None}) is False
    assert should_auto_revise({"terminal_state": "completed"}) is False
    assert should_auto_revise({"terminal_state": "aborted"}) is False
    assert should_auto_revise({"terminal_state": "budget_exhausted"}) is False
    assert should_auto_revise({"terminal_state": "criterion_failed"}) is True
    assert should_auto_revise({"terminal_state": "revision_requested"}) is True


def test_revisable_states_constant() -> None:
    assert "criterion_failed" in REVISABLE_TERMINAL_STATES
    assert "revision_requested" in REVISABLE_TERMINAL_STATES
    assert "completed" not in REVISABLE_TERMINAL_STATES


def test_summarize_for_stage0_after_revision_request(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    request_spec_revision(handle, reason="hypothesis didn't match the data")

    summary = summarize_for_stage0(handle.root)
    assert "rev-loop-fixture_rev1" in summary
    assert "revision_requested" in summary
    assert "hypothesis didn't match" in summary
    assert "## Question" in summary
    assert "Does the model X?" in summary
    # Final line nudges the agent toward setting parent_spec_id explicitly.
    assert "parent_spec_id" in summary
    assert "prior_results_ref" in summary


def test_summarize_after_failed_criterion(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    out = compute_and_commit_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="acc-001",
        inputs={"predictions": [0, 0, 1, 0], "labels": [1, 1, 1, 1]},
        split="dev",
        threshold=0.9,
        comparator=">=",
        criterion_id="c1",
    )
    assert out["criterion_record"]["verdict"] == "fail"
    state = read_state(handle.state_path)
    assert state.terminal_state.value == "criterion_failed"

    summary = summarize_for_stage0(handle.root)
    assert "criterion_failed" in summary
    assert "FAIL" in summary
    assert "c1" in summary
    assert "Findings on disk" in summary


def test_summarize_handles_missing_files(tmp_path: Path) -> None:
    """If the run dir is partial, summarize_for_stage0 must not crash."""
    bare = tmp_path / "empty_rev1"
    bare.mkdir()
    out = summarize_for_stage0(bare)
    # No claims, no criteria, no log — but the function still returns a string.
    assert isinstance(out, str)
    assert "Prior run summary" in out


def test_summarize_includes_log_tail(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    handle.narrative_path.write_text(
        "# Investigation Log\n\n"
        "Stage 0: ran black-box probe.\n"
        "Stage 1: tried activation patching at L9H9.\n"
        "Concluded: hypothesis was wrong; L10H7 dominates.\n"
    )
    request_spec_revision(handle, reason="L10H7 not L9H9")
    summary = summarize_for_stage0(handle.root)
    assert "L10H7 dominates" in summary


def test_summarize_truncates_findings_list(tmp_path: Path) -> None:
    """Many findings → only the first 10 are listed inline."""
    handle = init_run(_spec(), runs_root=tmp_path)
    findings = handle.findings_dir / "stage_0_black_box"
    findings.mkdir(parents=True)
    for i in range(15):
        (findings / f"behavioral_b{i:02d}.json").write_text("{}")
    summary = summarize_for_stage0(handle.root)
    assert "and 5 more" in summary
