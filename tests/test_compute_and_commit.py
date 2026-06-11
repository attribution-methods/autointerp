"""Tests for the compute_and_commit_metric one-shot wrapper."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from autointerp.pipelines.investigation import init_run
from autointerp.pipelines.investigation.metrics import compute_and_commit_metric
from autointerp.pipelines.investigation.tools import create_investigation_tools
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


def _spec_with_criterion() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="ccm-fixture",
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
        ],
        success_criteria=[
            Criterion(
                criterion_id="dev-acc",
                description="dev accuracy",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.5,
                on_split="dev",
            ),
        ],
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=_now()),
    )


def test_compute_and_commit_writes_finding(tmp_path: Path) -> None:
    handle = init_run(_spec_with_criterion(), runs_root=tmp_path)
    out = compute_and_commit_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="acc-001",
        inputs={"predictions": [1, 1, 0, 1], "labels": [1, 1, 1, 1]},
        split="dev",
        threshold=0.5,
        comparator=">=",
    )
    assert out["metric_result"]["value"] == 0.75
    ref = out["artifact_ref"]
    assert ref["kind"] == "MetricResult"
    assert ref["split"] == "dev"
    assert (handle.root / ref["relpath"]).exists()
    # No criterion evaluated when criterion_id not given.
    assert out["criterion_record"] is None


def test_compute_and_commit_with_criterion(tmp_path: Path) -> None:
    handle = init_run(_spec_with_criterion(), runs_root=tmp_path)
    out = compute_and_commit_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="acc-001",
        inputs={"predictions": [1, 1, 0, 1], "labels": [1, 1, 1, 1]},
        split="dev",
        threshold=0.5,
        comparator=">=",
        criterion_id="dev-acc",
    )
    rec = out["criterion_record"]
    assert rec is not None
    assert rec["criterion_id"] == "dev-acc"
    assert rec["verdict"] == "pass"
    assert rec["value"] == 0.75


def test_compute_and_commit_failed_criterion_terminates_run(tmp_path: Path) -> None:
    handle = init_run(_spec_with_criterion(), runs_root=tmp_path)
    out = compute_and_commit_metric(
        handle,
        metric=MetricName.ACCURACY,
        metric_id="acc-001",
        inputs={"predictions": [0, 0, 0, 1], "labels": [1, 1, 1, 1]},
        split="dev",
        threshold=0.5,
        comparator=">=",
        criterion_id="dev-acc",
    )
    assert out["criterion_record"]["verdict"] == "fail"
    # The criterion gate flips the run to CRITERION_FAILED.
    from autointerp.pipelines.investigation.state import TerminalState, read_state

    state = read_state(handle.state_path)
    assert state.terminal_state is TerminalState.CRITERION_FAILED


def test_tier2_tool_registered(tmp_path: Path) -> None:
    handle = init_run(_spec_with_criterion(), runs_root=tmp_path)
    tools = create_investigation_tools(handle)
    names = [t.name for t in tools]
    # Both the new wrapper and the original primitives must be there.
    assert "compute_and_commit_metric" in names
    assert "compute_metric" in names
    assert "commit_artifact" in names
    # Ergonomic wrapper is listed first to bias the agent toward using it.
    assert names.index("compute_and_commit_metric") < names.index("compute_metric")


def test_tier2_tool_round_trip(tmp_path: Path) -> None:
    handle = init_run(_spec_with_criterion(), runs_root=tmp_path)
    tools = {t.name: t for t in create_investigation_tools(handle)}
    tool = tools["compute_and_commit_metric"]

    out_str, ok = asyncio.run(
        tool.handler(
            {
                "metric": "accuracy",
                "metric_id": "acc-001",
                "inputs": {"predictions": [1, 1, 1, 1], "labels": [1, 1, 1, 1]},
                "split": "dev",
                "threshold": 0.5,
                "comparator": ">=",
                "criterion_id": "dev-acc",
            }
        )
    )
    assert ok, out_str
    parsed = json.loads(out_str)
    assert parsed["metric_result"]["value"] == 1.0
    assert parsed["criterion_record"]["verdict"] == "pass"


def test_tier2_tool_validates_required_args(tmp_path: Path) -> None:
    handle = init_run(_spec_with_criterion(), runs_root=tmp_path)
    tools = {t.name: t for t in create_investigation_tools(handle)}
    tool = tools["compute_and_commit_metric"]
    msg, ok = asyncio.run(tool.handler({"metric": "accuracy"}))
    assert ok is False
    assert "metric_id" in msg
