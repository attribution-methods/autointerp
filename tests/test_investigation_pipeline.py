"""End-to-end tests for the investigation pipeline gates.

Covers criteria.evaluate_criterion, guards (abort_if + budget), stages, the
revision exit, and final report assembly. The agent is *scripted* — i.e.
tools are called directly without an LLM — so these are deterministic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import (
    ArtifactGateError,
    CriterionGateError,
    GuardError,
    MetricRegistryError,
    RevisionGateError,
    StageGateError,
    advance_stage,
    assemble_report,
    check_abort_predicates,
    commit_artifact,
    compute_metric,
    current_stage_view,
    enforce_budget,
    evaluate_criterion,
    init_run,
    render_progress,
    request_spec_revision,
    write_report,
)
from autointerp.pipelines.investigation.state import (
    StageStatus,
    TerminalState,
    Verdict,
    read_state,
)
from autointerp.schemas import (
    BehaviorSpec,
    BehavioralFinding,
    ChatMessage,
    EvidenceStrength,
    PromptBatch,
    PromptCase,
)
from autointerp.spec import (
    AbortPredicate,
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec(
    *,
    abort_if: list | None = None,
    budget: Budget | None = None,
    criteria_split: str = "dev",
) -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="pipeline-fixture",
        revision=1,
        question="q",
        hypothesis="h",
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
            StageSpec(
                stage=InvestigationStage.VALIDATION,
                pattern=PatternId.BLACKBOX_THEN_PATCHING,
                tools=[ToolName.ACTIVATION_PATCHING],
                metrics=[MetricName.FAITHFULNESS],
            ),
        ],
        success_criteria=[
            Criterion(
                criterion_id="behavioral-sanity",
                description="d",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.5,
                on_split=criteria_split,
            ),
            Criterion(
                criterion_id="circuit-faithfulness",
                description="d",
                metric=MetricName.FAITHFULNESS,
                comparator=">=",
                threshold=0.7,
                on_split="heldout",
            ),
        ],
        abort_if=list(abort_if) if abort_if else [],
        budget=budget or Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=_now()),
    )


def _commit_accuracy(handle, *, value: float, split: str = "dev", metric_id: str = "acc-1"):
    """Commit a dummy PromptBatch and a MetricResult of the requested accuracy."""
    n_correct = round(value * 10)
    payload, token = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id=metric_id,
        inputs={
            "predictions": [1] * n_correct + [0] * (10 - n_correct),
            "labels": [1] * 10,
        },
    )
    return commit_artifact(
        handle, "MetricResult", payload, split=split, provenance_token=token
    )


def _commit_faithfulness(handle, *, ratio: float, split: str = "heldout", metric_id: str = "f-1"):
    payload, token = compute_metric(
        handle,
        metric=MetricName.FAITHFULNESS,
        metric_id=metric_id,
        inputs={
            "circuit_metric": ratio,
            "full_model_metric": 1.0,
            "corrupted_metric": 0.0,
        },
    )
    return commit_artifact(
        handle, "MetricResult", payload, split=split, provenance_token=token
    )


# ---- criteria --------------------------------------------------------------


def test_evaluate_criterion_pass(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.8, split="dev")
    rec = evaluate_criterion(handle, "behavioral-sanity")
    assert rec.verdict is Verdict.PASS
    assert rec.value == 0.8


def test_evaluate_criterion_fail_terminates(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.3, split="dev")
    rec = evaluate_criterion(handle, "behavioral-sanity")
    assert rec.verdict is Verdict.FAIL
    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.CRITERION_FAILED


def test_evaluate_criterion_one_shot_caches(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.8, split="dev")
    rec1 = evaluate_criterion(handle, "behavioral-sanity")
    # Second call returns cached record even though state has not changed.
    rec2 = evaluate_criterion(handle, "behavioral-sanity")
    assert rec1 == rec2


def test_evaluate_criterion_unknown_id(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    with pytest.raises(CriterionGateError, match="unknown criterion_id"):
        evaluate_criterion(handle, "nope")


def test_split_disjointness_refuses_wrong_split(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    # Commit accuracy on dev; criterion behavioral-sanity wants on_split=dev — fine.
    # But circuit-faithfulness wants heldout. Commit faithfulness on dev split:
    _commit_faithfulness(handle, ratio=0.9, split="dev")
    with pytest.raises(CriterionGateError, match="no committed MetricResult matches"):
        evaluate_criterion(handle, "circuit-faithfulness")


def test_explicit_metric_result_ref(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    ref = _commit_accuracy(handle, value=0.7, split="dev", metric_id="acc-explicit")
    rec = evaluate_criterion(
        handle, "behavioral-sanity", metric_result_ref=ref.relpath
    )
    assert rec.verdict is Verdict.PASS


def test_explicit_ref_split_mismatch_rejected(tmp_path: Path) -> None:
    """The criterion wants heldout, the ref points at a dev metric — refuse."""
    handle = init_run(_spec(), runs_root=tmp_path)
    ref = _commit_faithfulness(handle, ratio=0.9, split="dev")
    with pytest.raises(CriterionGateError, match="split disjointness"):
        evaluate_criterion(
            handle, "circuit-faithfulness", metric_result_ref=ref.relpath
        )


def test_multiple_matching_metrics_require_explicit_ref(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.6, split="dev", metric_id="a")
    _commit_accuracy(handle, value=0.9, split="dev", metric_id="b")
    with pytest.raises(CriterionGateError, match="multiple"):
        evaluate_criterion(handle, "behavioral-sanity")


# ---- guards: abort predicates ---------------------------------------------


def test_abort_predicate_typed_trips(tmp_path: Path) -> None:
    pred = AbortPredicate(
        predicate_id="abort-acc-low",
        description="abort if accuracy < 0.75",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.75,
        on_stage=InvestigationStage.BLACK_BOX,
    )
    handle = init_run(_spec(abort_if=[pred]), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.5, split="dev")
    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.ABORTED
    assert state.abort_triggered is not None
    assert state.abort_triggered.predicate_id == "abort-acc-low"


def test_abort_predicate_does_not_trip_when_above_threshold(tmp_path: Path) -> None:
    pred = AbortPredicate(
        predicate_id="abort-acc-low",
        description="d",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.75,
    )
    handle = init_run(_spec(abort_if=[pred]), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.9, split="dev")
    state = read_state(handle.state_path)
    assert state.terminal_state is None
    assert state.abort_triggered is None


def test_legacy_string_abort_is_advisory(tmp_path: Path) -> None:
    """Free-text abort_if entries must NOT trigger anything mechanical."""
    handle = init_run(
        _spec(abort_if=["accuracy too low — abort"]), runs_root=tmp_path
    )
    _commit_accuracy(handle, value=0.0, split="dev")
    state = read_state(handle.state_path)
    assert state.terminal_state is None  # advisory only


def test_abort_predicate_on_stage_filter(tmp_path: Path) -> None:
    """Predicate with on_stage=VALIDATION should not trip from a BLACK_BOX metric."""
    pred = AbortPredicate(
        predicate_id="abort-faith-low",
        description="d",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.99,
        on_stage=InvestigationStage.VALIDATION,
    )
    handle = init_run(_spec(abort_if=[pred]), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.5, split="dev")
    state = read_state(handle.state_path)
    assert state.terminal_state is None


def test_check_abort_idempotent(tmp_path: Path) -> None:
    pred = AbortPredicate(
        predicate_id="p",
        description="d",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.75,
    )
    handle = init_run(_spec(abort_if=[pred]), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.5, split="dev")
    rec1 = check_abort_predicates(handle)
    rec2 = check_abort_predicates(handle)
    assert rec1 == rec2


# ---- guards: budget --------------------------------------------------------


def test_budget_max_tool_calls_trips(tmp_path: Path) -> None:
    handle = init_run(_spec(budget=Budget(max_tool_calls=2)), runs_root=tmp_path)
    # 1 commit + 1 compute consumes 2 calls; the 3rd should be rejected.
    payload, token = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="m1",
        inputs={"predictions": [1], "labels": [1]},
    )
    commit_artifact(handle, "MetricResult", payload, split="dev", provenance_token=token)
    with pytest.raises(MetricRegistryError, match="budget exhausted"):
        compute_metric(
            handle,
            metric=MetricName.ACCURACY,
            metric_id="m2",
            inputs={"predictions": [1], "labels": [1]},
        )
    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.BUDGET_EXHAUSTED


def test_enforce_budget_no_op_when_unset(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    enforce_budget(handle)  # must not raise


# ---- stages ---------------------------------------------------------------


def test_advance_stage_refuses_without_metric(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    with pytest.raises(StageGateError, match="has not committed"):
        advance_stage(handle)


def test_advance_stage_after_metric(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.8, split="dev")
    out = advance_stage(handle)
    assert out["advanced_from"] == 0
    assert out["current_stage_idx"] == 1
    state = read_state(handle.state_path)
    assert state.stage_status["0"].status is StageStatus.COMPLETED
    assert state.stage_status["0"].ended_at is not None


def test_current_stage_view(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    view = current_stage_view(handle)
    assert view["stage_idx"] == 0
    assert view["stage"] == "black_box"
    assert view["metrics"] == ["accuracy"]
    assert view["committed_metric_names"] == []
    _commit_accuracy(handle, value=0.8, split="dev")
    view2 = current_stage_view(handle)
    assert "accuracy" in view2["committed_metric_names"]


def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)

    # Stage 0: commit a PromptBatch + accuracy, evaluate the dev criterion, advance.
    batch = PromptBatch(
        batch_id="dev-batch",
        behavior_id="b",
        cases=[PromptCase(prompt_id="p1", messages=[ChatMessage(role="user", content="hi")])],
        split="dev",
    )
    commit_artifact(handle, "PromptBatch", batch)
    _commit_accuracy(handle, value=0.8, split="dev")
    rec1 = evaluate_criterion(handle, "behavioral-sanity")
    assert rec1.verdict is Verdict.PASS
    advance_stage(handle)

    # Stage 1: commit faithfulness on heldout, evaluate the heldout criterion, advance.
    heldout = PromptBatch(
        batch_id="heldout-batch",
        behavior_id="b",
        cases=[PromptCase(prompt_id="p2", messages=[ChatMessage(role="user", content="bye")])],
        split="heldout",
    )
    commit_artifact(handle, "PromptBatch", heldout)
    _commit_faithfulness(handle, ratio=0.9, split="heldout")
    rec2 = evaluate_criterion(handle, "circuit-faithfulness")
    assert rec2.verdict is Verdict.PASS
    advance_stage(handle)

    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.COMPLETED
    assert set(state.criteria_evaluated) == {"behavioral-sanity", "circuit-faithfulness"}

    report_path = write_report(handle)
    assert report_path.exists()
    blob = json.loads(report_path.read_text())
    assert blob["report_id"] == "pipeline-fixture_rev1"
    assert blob["metadata"]["terminal_state"] == "completed"
    assert blob["metadata"]["criteria_evaluated"]["behavioral-sanity"]["verdict"] == "pass"


def test_advance_final_stage_refuses_with_unevaluated_criteria(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.8, split="dev")
    advance_stage(handle)
    _commit_faithfulness(handle, ratio=0.9, split="heldout")
    # Did not evaluate either criterion.
    with pytest.raises(StageGateError, match="unevaluated"):
        advance_stage(handle)


def test_advance_final_stage_rolls_back_on_unevaluated(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.8, split="dev")
    advance_stage(handle)
    _commit_faithfulness(handle, ratio=0.9, split="heldout")
    with pytest.raises(StageGateError):
        advance_stage(handle)
    state = read_state(handle.state_path)
    # Still on stage 1, status IN_PROGRESS, terminal_state still None.
    assert state.current_stage_idx == 1
    assert state.terminal_state is None
    assert state.stage_status["1"].status is StageStatus.IN_PROGRESS


# ---- revision exit --------------------------------------------------------


def test_request_spec_revision(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    req = request_spec_revision(handle, reason="hypothesis no longer fits", prior_results_ref="run.json")
    assert req.reason.startswith("hypothesis")
    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.REVISION_REQUESTED


def test_request_revision_after_criterion_failure(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.1, split="dev")
    evaluate_criterion(handle, "behavioral-sanity")  # fails -> CRITERION_FAILED
    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.CRITERION_FAILED
    # Revision is still allowed from CRITERION_FAILED.
    req = request_spec_revision(handle, reason="threshold was wrong")
    state2 = read_state(handle.state_path)
    assert state2.terminal_state is TerminalState.REVISION_REQUESTED
    assert req.reason == "threshold was wrong"


def test_request_revision_blocked_after_completed(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    state = read_state(handle.state_path)
    state.terminal_state = TerminalState.COMPLETED
    from autointerp.pipelines.investigation.state import write_state
    write_state(handle.state_path, state)
    with pytest.raises(RevisionGateError, match="terminal"):
        request_spec_revision(handle, reason="too late")


def test_request_revision_idempotent(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    a = request_spec_revision(handle, reason="reason A")
    b = request_spec_revision(handle, reason="reason B")
    assert a == b  # second call returns the first record


# ---- guards interaction with terminal state -------------------------------


def test_metric_compute_blocked_after_abort(tmp_path: Path) -> None:
    pred = AbortPredicate(
        predicate_id="p",
        description="d",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.75,
    )
    handle = init_run(_spec(abort_if=[pred]), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.5, split="dev")
    with pytest.raises(MetricRegistryError, match="terminal"):
        compute_metric(
            handle,
            metric=MetricName.ACCURACY,
            metric_id="m2",
            inputs={"predictions": [1], "labels": [1]},
        )


def test_commit_blocked_after_abort(tmp_path: Path) -> None:
    pred = AbortPredicate(
        predicate_id="p",
        description="d",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.75,
    )
    handle = init_run(_spec(abort_if=[pred]), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.5, split="dev")
    finding = BehavioralFinding(finding_id="f", behavior_id="b", summary="s")
    with pytest.raises(ArtifactGateError, match="terminal"):
        commit_artifact(handle, "BehavioralFinding", finding, split="dev")


# ---- report assembly ------------------------------------------------------


def test_assemble_report_minimal(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.6, split="dev")
    rec = evaluate_criterion(handle, "behavioral-sanity")
    assert rec.verdict is Verdict.PASS
    report = assemble_report(handle)
    assert report.report_id == "pipeline-fixture_rev1"
    assert any("PASS criterion 'behavioral-sanity'" in c for c in report.claims)


def test_guarderror_for_post_terminal_budget_call(tmp_path: Path) -> None:
    handle = init_run(_spec(budget=Budget(max_tool_calls=1)), runs_root=tmp_path)
    payload, token = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="m1",
        inputs={"predictions": [1], "labels": [1]},
    )
    # First call already consumed the only allowed tool call.
    with pytest.raises(ArtifactGateError, match="budget exhausted"):
        commit_artifact(handle, "MetricResult", payload, split="dev", provenance_token=token)
    # Subsequent enforce_budget calls raise too.
    with pytest.raises(GuardError, match="budget exhausted"):
        enforce_budget(handle)


# ---- inconclusive verdict -------------------------------------------------


def test_evaluate_criterion_inconclusive_does_not_terminate(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    # Value 0.3 would FAIL the >=0.5 criterion, but the agent declares the
    # evidence inconclusive instead. The run must NOT terminate.
    _commit_accuracy(handle, value=0.3, split="dev")
    rec = evaluate_criterion(
        handle,
        "behavioral-sanity",
        inconclusive_reason="only 3 usable samples after dedup",
    )
    assert rec.verdict is Verdict.INCONCLUSIVE
    assert rec.inconclusive_reason == "only 3 usable samples after dedup"
    assert rec.value == 0.3  # value still recorded for the report
    state = read_state(handle.state_path)
    assert state.terminal_state is None
    assert "behavioral-sanity" in state.criteria_evaluated


def test_inconclusive_overrides_a_passing_value(tmp_path: Path) -> None:
    """Reason wins: a value that would PASS is still INCONCLUSIVE if flagged."""
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.9, split="dev")
    rec = evaluate_criterion(
        handle, "behavioral-sanity", inconclusive_reason="dev set leaked into train"
    )
    assert rec.verdict is Verdict.INCONCLUSIVE
    state = read_state(handle.state_path)
    assert state.terminal_state is None


def test_blank_inconclusive_reason_falls_back_to_passfail(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.8, split="dev")
    rec = evaluate_criterion(handle, "behavioral-sanity", inconclusive_reason="   ")
    assert rec.verdict is Verdict.PASS
    assert rec.inconclusive_reason is None


def test_inconclusive_criterion_counts_for_final_advance(tmp_path: Path) -> None:
    """A run can terminate COMPLETED with an inconclusive criterion."""
    handle = init_run(_spec(), runs_root=tmp_path)
    batch = PromptBatch(
        batch_id="dev-batch",
        behavior_id="b",
        cases=[PromptCase(prompt_id="p1", messages=[ChatMessage(role="user", content="hi")])],
        split="dev",
    )
    commit_artifact(handle, "PromptBatch", batch)
    _commit_accuracy(handle, value=0.8, split="dev")
    evaluate_criterion(handle, "behavioral-sanity")
    advance_stage(handle)

    heldout = PromptBatch(
        batch_id="heldout-batch",
        behavior_id="b",
        cases=[PromptCase(prompt_id="p2", messages=[ChatMessage(role="user", content="bye")])],
        split="heldout",
    )
    commit_artifact(handle, "PromptBatch", heldout)
    _commit_faithfulness(handle, ratio=0.9, split="heldout")
    rec = evaluate_criterion(
        handle, "circuit-faithfulness", inconclusive_reason="patch set too small"
    )
    assert rec.verdict is Verdict.INCONCLUSIVE
    advance_stage(handle)

    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.COMPLETED


# ---- progress digest ------------------------------------------------------


def test_progress_md_written_on_init(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    assert handle.progress_path.exists()
    text = handle.progress_path.read_text()
    assert "pipeline-fixture_rev1" in text
    assert "Status: in_progress" in text
    # Unevaluated criteria show as pending.
    assert "pending" in text
    assert "behavioral-sanity" in text


def test_render_progress_reflects_verdicts(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.8, split="dev")
    evaluate_criterion(handle, "behavioral-sanity")
    text = render_progress(handle)
    assert "PASS" in text
    assert "behavioral-sanity" in text
    # The other criterion is still pending.
    assert "pending" in text
    assert "circuit-faithfulness" in text


def test_progress_md_refreshed_on_inconclusive(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.3, split="dev")
    evaluate_criterion(
        handle, "behavioral-sanity", inconclusive_reason="too few samples"
    )
    text = handle.progress_path.read_text()
    assert "INCONCLUSIVE" in text
    assert "too few samples" in text


def test_progress_md_shows_terminal_after_fail(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    _commit_accuracy(handle, value=0.1, split="dev")
    evaluate_criterion(handle, "behavioral-sanity")  # FAIL -> terminal
    text = handle.progress_path.read_text()
    assert "Status: criterion_failed" in text
    assert "FAIL" in text
