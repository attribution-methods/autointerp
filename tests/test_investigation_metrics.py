"""Tests for the canonical metric registry."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import (
    METRIC_REGISTRY,
    MetricRegistryError,
    commit_artifact,
    compute_metric,
    init_run,
)
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _approved_spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="metrics-fixture",
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
            )
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
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=_now()),
    )


# ---- pure compute correctness ---------------------------------------------


def test_accuracy_basic() -> None:
    impl = METRIC_REGISTRY[MetricName.ACCURACY]
    assert impl.compute({"predictions": [1, 1, 0, 1], "labels": [1, 0, 0, 1]}) == 0.75


def test_logit_diff_basic() -> None:
    impl = METRIC_REGISTRY[MetricName.LOGIT_DIFF]
    v = impl.compute({"target_logits": [2.0, 3.0], "foil_logits": [1.0, 1.5]})
    assert math.isclose(v, ((2.0 - 1.0) + (3.0 - 1.5)) / 2)


def test_faithfulness_basic() -> None:
    impl = METRIC_REGISTRY[MetricName.FAITHFULNESS]
    v = impl.compute(
        {"circuit_metric": 0.8, "full_model_metric": 1.0, "corrupted_metric": 0.0}
    )
    assert math.isclose(v, 0.8)


def test_effect_size_unpaired() -> None:
    impl = METRIC_REGISTRY[MetricName.EFFECT_SIZE]
    # Two groups with mean diff 2.0 and pooled SD 1.0 → d = 2.0
    v = impl.compute({"group_a": [3.0, 4.0, 5.0], "group_b": [1.0, 2.0, 3.0]})
    assert math.isclose(v, 2.0)


def test_effect_size_paired() -> None:
    impl = METRIC_REGISTRY[MetricName.EFFECT_SIZE]
    # diffs = [2, 1, 0] → mean=1.0, sd=1.0 → d_z = 1.0
    v = impl.compute(
        {"group_a": [3.0, 4.0, 5.0], "group_b": [1.0, 3.0, 5.0], "paired": True}
    )
    assert math.isclose(v, 1.0)


def test_effect_size_zero_sd_raises() -> None:
    impl = METRIC_REGISTRY[MetricName.EFFECT_SIZE]
    with pytest.raises(MetricRegistryError, match="undefined"):
        impl.compute({"group_a": [1.0, 1.0, 1.0], "group_b": [2.0, 2.0, 2.0]})


def test_faithfulness_zero_gap_raises() -> None:
    impl = METRIC_REGISTRY[MetricName.FAITHFULNESS]
    with pytest.raises(MetricRegistryError, match="undefined"):
        impl.compute(
            {"circuit_metric": 0.5, "full_model_metric": 0.5, "corrupted_metric": 0.5}
        )


def test_accuracy_length_mismatch() -> None:
    impl = METRIC_REGISTRY[MetricName.ACCURACY]
    with pytest.raises(MetricRegistryError, match="lengths"):
        impl.compute({"predictions": [1, 0], "labels": [1]})


def test_accuracy_empty_rejected() -> None:
    impl = METRIC_REGISTRY[MetricName.ACCURACY]
    with pytest.raises(MetricRegistryError, match="empty"):
        impl.compute({"predictions": [], "labels": []})


def test_logit_diff_non_numeric_rejected() -> None:
    impl = METRIC_REGISTRY[MetricName.LOGIT_DIFF]
    with pytest.raises(MetricRegistryError, match="not a number"):
        impl.compute({"target_logits": ["a"], "foil_logits": [0.0]})


def test_missing_keys_rejected() -> None:
    impl = METRIC_REGISTRY[MetricName.LOGIT_DIFF]
    with pytest.raises(MetricRegistryError, match="missing required input keys"):
        impl.compute({"target_logits": [1.0]})


# ---- compute_metric public path -------------------------------------------


def test_compute_metric_round_trip(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    payload, token = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="acc-001",
        inputs={"predictions": [1, 1, 0, 1], "labels": [1, 0, 0, 1]},
        threshold=0.5,
        comparator=">=",
    )
    assert token.startswith("tok_")
    assert payload["metric_id"] == "acc-001"
    assert payload["value"] == 0.75
    assert payload["threshold"] == 0.5
    assert payload["passed"] is True
    assert payload["metadata"]["metric_family"] == "behavioral"
    assert payload["metadata"]["inputs_hash"].startswith("sha256:")

    state = read_state(handle.state_path)
    assert token in state.pending_provenance_tokens
    rec = state.pending_provenance_tokens[token]
    assert rec.metric == "accuracy"
    assert rec.metric_id == "acc-001"
    assert rec.value == 0.75
    assert state.budget_consumed.tool_calls == 1


def test_compute_then_commit_consumes_token(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    payload, token = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="acc-001",
        inputs={"predictions": [1, 1, 0], "labels": [1, 1, 0]},
    )
    ref = commit_artifact(
        handle, "MetricResult", payload, split="dev", provenance_token=token
    )
    assert ref.relpath == "findings/stage_0_black_box/metric_acc-001.json"
    state = read_state(handle.state_path)
    assert token not in state.pending_provenance_tokens
    assert state.provenance_tokens_consumed == 1


def test_compute_string_metric_name_accepted(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    payload, _tok = compute_metric(
        handle,
        metric="accuracy",
        metric_id="m1",
        inputs={"predictions": [1], "labels": [1]},
    )
    assert payload["value"] == 1.0


def test_compute_unknown_metric_string_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(MetricRegistryError, match="unknown metric"):
        compute_metric(
            handle, metric="bogus", metric_id="m1", inputs={}
        )


def test_compute_metric_without_canonical_impl(tmp_path: Path) -> None:
    """Closed vocabulary metrics that aren't in the registry yet are rejected."""
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(MetricRegistryError, match="no canonical implementation"):
        compute_metric(
            handle,
            metric=MetricName.COMPLETENESS,
            metric_id="m1",
            inputs={},
        )


def test_faithfulness_clipping_records_unclipped(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    # circuit beats clean → ratio > 1, clipped to 1.0
    payload, _tok = compute_metric(
        handle,
        metric=MetricName.FAITHFULNESS,
        metric_id="f1",
        inputs={
            "circuit_metric": 1.2,
            "full_model_metric": 1.0,
            "corrupted_metric": 0.0,
        },
    )
    assert payload["value"] == 1.0
    assert payload["metadata"]["unclipped_value"] == 1.2
    assert payload["metadata"]["clipped_to_range"] == [0.0, 1.0]


def test_threshold_pass_fail_recorded(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    payload, _tok = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="m1",
        inputs={"predictions": [1, 0], "labels": [1, 1]},
        threshold=0.8,
        comparator=">=",
    )
    assert payload["value"] == 0.5
    assert payload["passed"] is False


def test_terminal_state_blocks_compute(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    state = read_state(handle.state_path)
    state.terminal_state = TerminalState.COMPLETED
    write_state(handle.state_path, state)
    with pytest.raises(MetricRegistryError, match="terminal state"):
        compute_metric(
            handle,
            metric=MetricName.ACCURACY,
            metric_id="m1",
            inputs={"predictions": [1], "labels": [1]},
        )


def test_compute_metric_logs_jsonl_line(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _payload, token = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="m1",
        inputs={"predictions": [1], "labels": [1]},
    )
    lines = [
        json.loads(line) for line in handle.log_path.read_text().splitlines() if line
    ]
    assert any(
        e["tool"] == "compute_metric"
        and e["args"]["metric"] == "accuracy"
        and token in e["result_summary"]
        for e in lines
    )


def test_inputs_hash_is_stable(tmp_path: Path) -> None:
    """Same inputs in different key order hash identically."""
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    p1, _ = compute_metric(
        handle,
        metric=MetricName.LOGIT_DIFF,
        metric_id="ld-1",
        inputs={"target_logits": [1.0], "foil_logits": [0.0]},
    )
    p2, _ = compute_metric(
        handle,
        metric=MetricName.LOGIT_DIFF,
        metric_id="ld-2",
        inputs={"foil_logits": [0.0], "target_logits": [1.0]},
    )
    assert p1["metadata"]["inputs_hash"] == p2["metadata"]["inputs_hash"]


def test_token_value_locked_against_tampering(tmp_path: Path) -> None:
    """commit_artifact must reject if the agent edits MetricResult.value after compute."""
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    payload, token = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="m1",
        inputs={"predictions": [1, 1, 0], "labels": [1, 0, 0]},
    )
    payload["value"] = 0.99  # try to fudge
    from autointerp.pipelines.investigation import ArtifactGateError

    with pytest.raises(ArtifactGateError, match="value"):
        commit_artifact(
            handle, "MetricResult", payload, split="dev", provenance_token=token
        )


def test_metric_id_required(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(MetricRegistryError, match="metric_id"):
        compute_metric(
            handle,
            metric=MetricName.ACCURACY,
            metric_id="",
            inputs={"predictions": [1], "labels": [1]},
        )


def test_compute_two_metrics_produces_distinct_tokens(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _, t1 = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="a1",
        inputs={"predictions": [1], "labels": [1]},
    )
    _, t2 = compute_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="a2",
        inputs={"predictions": [1], "labels": [1]},
    )
    assert t1 != t2
    state = read_state(handle.state_path)
    assert t1 in state.pending_provenance_tokens
    assert t2 in state.pending_provenance_tokens
