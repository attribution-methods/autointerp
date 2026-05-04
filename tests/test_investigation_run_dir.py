"""Tests for the investigation pipeline run-dir + state scaffolding."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import RunHandle, init_run, load_run
from autointerp.pipelines.investigation.state import (
    BudgetConsumed,
    CriterionRecord,
    ProvenanceToken,
    RunState,
    StageStatus,
    TerminalState,
    initial_state,
    now_iso,
    read_state,
    write_state,
)
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
from autointerp.schemas import BehaviorSpec


def _make_approved_spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="round-trip-fixture",
        revision=1,
        question="Does the round-trip work?",
        hypothesis="Yes, if state.json is atomic.",
        phenomenon_id="custom",
        behavior=BehaviorSpec(
            behavior_id="round-trip-behavior",
            description="Trivial behavior used to round-trip the scaffolding.",
        ),
        model=ModelRef(model_id="fixture/model", provider="local"),
        dataset=DatasetSpec(
            dataset_id="fixture-dataset",
            source="generated",
            n_samples=10,
            split="dev",
            seed=42,
        ),
        contrast=ContrastSpec(
            contrast_id="fixture-contrast",
            positive_template="The {x}.",
            negative_template="A {x}.",
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
                description="Trivial sanity threshold.",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.5,
                on_split="dev",
            ),
        ],
        budget=Budget(max_tool_calls=10),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="test", approver_kind="agent", approved_at=now_iso()),
    )


def test_initial_state_round_trip(tmp_path: Path) -> None:
    state = initial_state(
        spec_id="x",
        spec_revision=1,
        n_stages=2,
        stage_names=["black_box", "validation"],
    )
    path = tmp_path / "state.json"
    write_state(path, state)
    assert path.exists()
    loaded = read_state(path)
    assert loaded == state
    assert loaded.run_id == "x_rev1"
    assert loaded.stage_status["0"].stage == "black_box"
    assert loaded.stage_status["1"].status is StageStatus.PENDING
    assert loaded.budget_consumed == BudgetConsumed()


def test_state_handles_optional_records(tmp_path: Path) -> None:
    state = initial_state("x", 1, 1, ["black_box"])
    state.criteria_evaluated["c1"] = CriterionRecord(
        passed=True,
        value=0.9,
        metric="accuracy",
        comparator=">=",
        threshold=0.8,
        metric_result_ref="findings/stage_0_black_box/metric_acc-001.json",
        evaluated_at=now_iso(),
    )
    state.pending_provenance_tokens["tok_abc"] = ProvenanceToken(
        metric="accuracy",
        metric_id="acc-001",
        value=0.9,
        inputs_hash="sha256:deadbeef",
        issued_at=now_iso(),
        stage_idx=0,
    )
    state.terminal_state = TerminalState.COMPLETED

    path = tmp_path / "state.json"
    write_state(path, state)
    loaded = read_state(path)
    assert loaded.criteria_evaluated["c1"].passed is True
    assert loaded.pending_provenance_tokens["tok_abc"].metric == "accuracy"
    assert loaded.terminal_state is TerminalState.COMPLETED


def test_state_rejects_unknown_fields(tmp_path: Path) -> None:
    raw = {
        "schema_version": 1,
        "spec_id": "x",
        "spec_revision": 1,
        "run_id": "x_rev1",
        "run_started_at": now_iso(),
        "current_stage_idx": 0,
        "stage_status": {},
        "criteria_evaluated": {},
        "budget_consumed": BudgetConsumed().model_dump(),
        "pending_provenance_tokens": {},
        "provenance_tokens_consumed": 0,
        "MYSTERY_FIELD": True,  # should be rejected
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(Exception):
        read_state(path)


def test_atomic_write_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = initial_state("x", 1, 1, ["black_box"])
    write_state(path, state)
    # No leftover tempfiles in the parent dir.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".state.")]
    assert leftovers == []


def test_init_run_scaffolds_layout(tmp_path: Path) -> None:
    spec = _make_approved_spec()
    handle = init_run(spec, runs_root=tmp_path)

    assert isinstance(handle, RunHandle)
    assert handle.run_id == "round-trip-fixture_rev1"
    assert handle.root == tmp_path / handle.run_id

    for d in (
        handle.prompt_batches_dir,
        handle.activations_dir,
        handle.generations_dir,
        handle.findings_dir,
        handle.scripts_dir,
        handle.scratch_dir,
    ):
        assert d.is_dir()

    assert handle.spec_path.exists()
    assert handle.state_path.exists()
    assert handle.log_path.exists()
    assert handle.narrative_path.exists()

    # spec.json frozen read-only (no user write bit)
    mode = handle.spec_path.stat().st_mode
    assert not (mode & stat.S_IWUSR)

    # state.json identity matches spec
    state = read_state(handle.state_path)
    assert state.spec_id == spec.spec_id
    assert state.spec_revision == spec.revision
    assert len(state.stage_status) == len(spec.stages)
    assert state.stage_status["0"].stage == "black_box"
    assert state.stage_status["1"].stage == "validation"


def test_init_run_refuses_unapproved_spec(tmp_path: Path) -> None:
    spec = _make_approved_spec().model_copy(
        update={"status": SpecStatus.DRAFT, "approval": None}
    )
    with pytest.raises(ValueError, match="approved"):
        init_run(spec, runs_root=tmp_path)


def test_init_run_refuses_existing_dir(tmp_path: Path) -> None:
    spec = _make_approved_spec()
    init_run(spec, runs_root=tmp_path)
    with pytest.raises(FileExistsError):
        init_run(spec, runs_root=tmp_path)


def test_load_run_resumes(tmp_path: Path) -> None:
    spec = _make_approved_spec()
    handle = init_run(spec, runs_root=tmp_path)

    # Mutate state as the gate would, then resume.
    state = read_state(handle.state_path)
    state.current_stage_idx = 1
    state.stage_status["0"].status = StageStatus.COMPLETED
    state.budget_consumed.tool_calls = 7
    write_state(handle.state_path, state)

    handle2, spec2, state2 = load_run(handle.root)
    assert handle2.run_id == handle.run_id
    assert spec2.spec_id == spec.spec_id
    assert state2.current_stage_idx == 1
    assert state2.stage_status["0"].status is StageStatus.COMPLETED
    assert state2.budget_consumed.tool_calls == 7


def test_load_run_rejects_mismatched_state(tmp_path: Path) -> None:
    spec = _make_approved_spec()
    handle = init_run(spec, runs_root=tmp_path)

    # Tamper state.json identity.
    state = read_state(handle.state_path)
    raw = state.model_dump()
    raw["spec_id"] = "different-spec"
    handle.state_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    handle.state_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="identity"):
        load_run(handle.root)


def test_load_run_reasserts_readonly_spec(tmp_path: Path) -> None:
    spec = _make_approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    # Loosen perms and confirm load_run re-freezes.
    handle.spec_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert os.access(handle.spec_path, os.W_OK)
    load_run(handle.root)
    assert not os.access(handle.spec_path, os.W_OK)


def test_run_handle_writable_roots_are_agent_only(tmp_path: Path) -> None:
    spec = _make_approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    roots = handle.writable_roots()
    assert handle.scripts_dir in roots
    assert handle.scratch_dir in roots
    assert handle.narrative_path in roots
    assert handle.spec_path not in roots
    assert handle.state_path not in roots
    assert handle.findings_dir not in roots


def test_state_round_trip_via_run_dir(tmp_path: Path) -> None:
    spec = _make_approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    state = read_state(handle.state_path)
    # All fields should be defaulted on a fresh run.
    assert state.terminal_state is None
    assert state.abort_triggered is None
    assert state.spec_revision_requested is None
    assert state.pending_provenance_tokens == {}
    assert state.provenance_tokens_consumed == 0
    # Re-write should be byte-stable for an unchanged state.
    before = handle.state_path.read_text()
    write_state(handle.state_path, state)
    after = handle.state_path.read_text()
    assert before == after


def test_run_state_atomic_overwrite(tmp_path: Path) -> None:
    """Overwriting state.json must not leave an old or partial file behind."""
    path = tmp_path / "state.json"
    s1 = initial_state("x", 1, 1, ["black_box"])
    write_state(path, s1)
    s2 = s1.model_copy(update={"current_stage_idx": 0})
    s2.stage_status["0"].status = StageStatus.IN_PROGRESS
    write_state(path, s2)
    loaded = read_state(path)
    assert loaded.stage_status["0"].status is StageStatus.IN_PROGRESS
    # No leftover temp files.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".state.")]
    assert leftovers == []
