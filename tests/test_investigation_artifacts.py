"""Tests for the commit_artifact gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import (
    ArtifactGateError,
    commit_artifact,
    init_run,
)
from autointerp.pipelines.investigation.state import (
    ProvenanceToken,
    StageStatus,
    TerminalState,
    now_iso,
    read_state,
    write_state,
)
from autointerp.schemas import (
    ActivationCacheRef,
    BehavioralFinding,
    BehaviorSpec,
    ChatMessage,
    EvidenceStrength,
    MetricResult,
    ModelRef,
    PromptBatch,
    PromptCase,
)
from autointerp.schemas import ModelRef as SchemaModelRef
from autointerp.spec import (
    Approval,
    Budget,
    ContrastSpec,
    Criterion,
    DatasetSpec,
    InvestigationSpec,
    InvestigationStage,
    MetricName,
    PatternId,
    SpecStatus,
    StageSpec,
    ToolName,
)


def _approved_spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="artifacts-fixture",
        revision=1,
        question="Q",
        hypothesis="H",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="fixture/model", provider="local"),
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
                criterion_id="c1",
                description="d",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.5,
                on_split="dev",
            )
        ],
        budget=Budget(max_tool_calls=10),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=now_iso()),
    )


def _make_prompt_batch(batch_id: str = "b1", split: str = "dev") -> PromptBatch:
    return PromptBatch(
        batch_id=batch_id,
        behavior_id="b",
        cases=[
            PromptCase(prompt_id="p1", messages=[ChatMessage(role="user", content="hi")]),
        ],
        split=split,
    )


def _seed_token(
    handle, *, token: str, metric: str, metric_id: str, value: float, stage_idx: int = 0
):
    state = read_state(handle.state_path)
    state.pending_provenance_tokens[token] = ProvenanceToken(
        metric=metric,
        metric_id=metric_id,
        value=value,
        inputs_hash="sha256:test",
        issued_at=now_iso(),
        stage_idx=stage_idx,
    )
    write_state(handle.state_path, state)


# ---- happy paths -----------------------------------------------------------


def test_commit_prompt_batch_roundtrip(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    ref = commit_artifact(handle, "PromptBatch", batch)
    assert ref.kind == "PromptBatch"
    assert ref.artifact_id == "b1"
    assert ref.relpath == "prompt_batches/b1.json"
    assert ref.path.exists()
    payload = json.loads(ref.path.read_text())
    assert payload["batch_id"] == "b1"
    assert payload["metadata"]["_provenance"]["stage_idx"] == 0
    assert payload["metadata"]["_provenance"]["split"] == "dev"

    state = read_state(handle.state_path)
    assert state.stage_status["0"].status is StageStatus.IN_PROGRESS
    assert state.stage_status["0"].artifact_refs == ["prompt_batches/b1.json"]
    assert state.budget_consumed.tool_calls == 1


def test_commit_dict_payload_validated(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    ref = commit_artifact(
        handle,
        "PromptBatch",
        {
            "batch_id": "b2",
            "behavior_id": "b",
            "cases": [{"prompt_id": "p1", "messages": [{"role": "user", "content": "hi"}]}],
            "split": "dev",
        },
    )
    assert ref.path.exists()


def test_commit_finding_into_stage_dir(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    finding = BehavioralFinding(
        finding_id="signal-check",
        behavior_id="b",
        summary="signal exists",
        evidence_strength=EvidenceStrength.MODERATE,
    )
    ref = commit_artifact(handle, "BehavioralFinding", finding, split="dev")
    assert ref.relpath == "findings/stage_0_black_box/behavioral_signal-check.json"
    assert ref.path.exists()


def test_commit_activation_cache_ref(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    cache = ActivationCacheRef(
        cache_id="cache-1",
        model=SchemaModelRef(model_id="m"),
        source_path="activations/cache-1/",
        prompt_batch_id="b1",
        layers=[0, 1, 2],
    )
    ref = commit_artifact(handle, "ActivationCacheRef", cache, split="dev")
    assert ref.relpath == "activations/cache-1.json"


def test_commit_metric_with_valid_token(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-001", value=0.83)
    ref = commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")
    assert ref.relpath == "findings/stage_0_black_box/metric_acc-001.json"
    state = read_state(handle.state_path)
    assert "tok1" not in state.pending_provenance_tokens
    assert state.provenance_tokens_consumed == 1


# ---- gate failures ---------------------------------------------------------


def test_unknown_kind_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(ArtifactGateError, match="Unknown artifact kind"):
        commit_artifact(handle, "Whatever", {}, split="dev")


def test_invalid_payload_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(ArtifactGateError, match="failed validation"):
        commit_artifact(handle, "PromptBatch", {"batch_id": "x"})


def test_split_required_for_non_promptbatch(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    finding = BehavioralFinding(finding_id="f", behavior_id="b", summary="s")
    with pytest.raises(ArtifactGateError, match="split is required"):
        commit_artifact(handle, "BehavioralFinding", finding)


def test_promptbatch_split_arg_must_match(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch(split="dev")
    with pytest.raises(ArtifactGateError, match="split mismatch"):
        commit_artifact(handle, "PromptBatch", batch, split="heldout")


def test_overwrite_refused(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    commit_artifact(handle, "PromptBatch", batch)
    with pytest.raises(ArtifactGateError, match="append-only"):
        commit_artifact(handle, "PromptBatch", batch)


def test_metric_requires_token(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    metric = MetricResult(metric_id="acc-001", value=0.83)
    with pytest.raises(ArtifactGateError, match="provenance_token"):
        commit_artifact(handle, "MetricResult", metric, split="dev")


def test_token_not_reusable(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-001", value=0.83)
    commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")
    metric2 = MetricResult(metric_id="acc-002", value=0.83)
    with pytest.raises(ArtifactGateError, match="unknown or already consumed"):
        commit_artifact(handle, "MetricResult", metric2, split="dev", provenance_token="tok1")


def test_token_metric_id_mismatch(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-OTHER", value=0.83)
    with pytest.raises(ArtifactGateError, match="metric_id"):
        commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")


def test_token_value_mismatch_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-001", value=0.99)
    with pytest.raises(ArtifactGateError, match="value"):
        commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")


def test_token_stage_mismatch_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(
        handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83, stage_idx=1
    )
    metric = MetricResult(metric_id="acc-001", value=0.83)
    with pytest.raises(ArtifactGateError, match="stage_idx"):
        commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")


def test_token_only_for_metric_result(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    with pytest.raises(ArtifactGateError, match="only valid for MetricResult"):
        commit_artifact(handle, "PromptBatch", batch, provenance_token="tok-anything")


def test_terminal_state_blocks_commits(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    state = read_state(handle.state_path)
    state.terminal_state = TerminalState.COMPLETED
    write_state(handle.state_path, state)
    with pytest.raises(ArtifactGateError, match="terminal state"):
        commit_artifact(handle, "PromptBatch", _make_prompt_batch())


def test_reserved_provenance_metadata_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    batch = batch.model_copy(update={"metadata": {"_provenance": {"hijack": True}}})
    with pytest.raises(ArtifactGateError, match="_provenance"):
        commit_artifact(handle, "PromptBatch", batch)


def test_log_jsonl_appended(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    commit_artifact(handle, "PromptBatch", _make_prompt_batch())
    lines = [
        json.loads(line)
        for line in handle.log_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["tool"] == "commit_artifact"
    assert entry["args"]["kind"] == "PromptBatch"
    assert entry["ok"] is True
    assert entry["budget_after"]["tool_calls"] == 1


def test_in_progress_started_at_set_once(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    commit_artifact(handle, "PromptBatch", _make_prompt_batch("b1"))
    state1 = read_state(handle.state_path)
    started1 = state1.stage_status["0"].started_at
    assert started1 is not None
    commit_artifact(handle, "PromptBatch", _make_prompt_batch("b2"))
    state2 = read_state(handle.state_path)
    assert state2.stage_status["0"].started_at == started1
    assert state2.stage_status["0"].status is StageStatus.IN_PROGRESS
