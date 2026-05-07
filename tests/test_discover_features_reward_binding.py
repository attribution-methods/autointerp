"""Tests for the reward-metric binding in `discover_features`.

The discovery sub-agent must hill-climb on a metric the user pre-registered
in `spec.stages[idx].metrics`. The Tier-2 handler enforces that:

- if the stage lists exactly one metric, the handler defaults to it;
- if the stage lists multiple metrics, the handler refuses without an
  explicit `reward_metric`;
- in either shape, an off-spec `reward_metric` is refused with a clear
  message pointing the agent at `request_spec_revision`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import init_run
from autointerp.pipelines.investigation.state import (
    StageStatus,
    now_iso,
    read_state,
    write_state,
)
from autointerp.pipelines.investigation.tools import create_investigation_tools
from autointerp.schemas import BehaviorSpec
from autointerp.spec import (
    Approval,
    Budget,
    ContrastSpec,
    Criterion,
    DatasetSpec,
    DiscoveryConfig,
    InvestigationSpec,
    InvestigationStage,
    MetricName,
    ModelRef,
    PatternId,
    SpecStatus,
    StageSpec,
    ToolName,
)


def _make_spec(*, discovery_metrics: list[MetricName]) -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="reward-binding-fixture",
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
            positive_template="x -> y",
            negative_template="x' -> y",
        ),
        stages=[
            StageSpec(
                stage=InvestigationStage.FEATURE_DISCOVERY,
                pattern=PatternId.BLACKBOX_THEN_PATCHING,
                tools=[ToolName.DISCOVER_FEATURES],
                metrics=discovery_metrics,
                discovery=DiscoveryConfig(
                    substrate="components",
                    component_kinds=["attn_head"],
                ),
            ),
        ],
        success_criteria=[
            Criterion(
                criterion_id="c0",
                description="check the discovery reward",
                metric=discovery_metrics[0],
                comparator=">=",
                threshold=0.5,
                on_split="dev",
            ),
        ],
        abort_if=[],
        budget=Budget(),
        risks=[],
        status=SpecStatus.APPROVED,
        approval=Approval(
            approver="user",
            approver_kind="human",
            approved_at=datetime.now(timezone.utc),
            notes="test fixture",
        ),
    )


def _bootstrap_run(tmp_path: Path, spec: InvestigationSpec):
    """Init a run dir and bump the active stage to feature_discovery (idx 0)."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    state = read_state(handle.state_path)
    state.stage_status["0"].status = StageStatus.IN_PROGRESS
    state.stage_status["0"].started_at = now_iso()
    write_state(handle.state_path, state)
    return handle


def _discover_tool(handle):
    tools = {t.name: t for t in create_investigation_tools(handle)}
    return tools["discover_features"]


# ---------------------------------------------------------------------------
# unique-metric default
# ---------------------------------------------------------------------------


def test_unique_metric_is_used_by_default(tmp_path: Path):
    handle = _bootstrap_run(
        tmp_path, _make_spec(discovery_metrics=[MetricName.COMBINED_AUC_K])
    )
    tool = _discover_tool(handle)
    out, ok = asyncio.run(tool.handler({"task": "demo", "dry_run": True}))
    assert ok, out
    assert "combined_auc_k" in out  # surfaced in the result payload


def test_off_spec_reward_refused_under_unique_stage(tmp_path: Path):
    handle = _bootstrap_run(
        tmp_path, _make_spec(discovery_metrics=[MetricName.COMBINED_AUC_K])
    )
    tool = _discover_tool(handle)
    out, ok = asyncio.run(
        tool.handler(
            {"task": "demo", "reward_metric": "patch_effect_recovery", "dry_run": True}
        )
    )
    assert not ok
    assert "not in the stage's pre-registered metrics" in out
    assert "patch_effect_recovery" in out
    assert "request_spec_revision" in out  # actionable next step surfaced


# ---------------------------------------------------------------------------
# multi-metric stages — explicit picks required
# ---------------------------------------------------------------------------


def test_ambiguous_stage_requires_explicit_pick(tmp_path: Path):
    handle = _bootstrap_run(
        tmp_path,
        _make_spec(
            discovery_metrics=[MetricName.COMBINED_AUC_K, MetricName.PATCH_EFFECT_RECOVERY]
        ),
    )
    tool = _discover_tool(handle)
    out, ok = asyncio.run(tool.handler({"task": "demo", "dry_run": True}))
    assert not ok
    assert "reward_metric is required" in out
    # both options should be surfaced so the agent knows what's allowed.
    assert "combined_auc_k" in out and "patch_effect_recovery" in out


def test_explicit_in_spec_metric_is_accepted(tmp_path: Path):
    handle = _bootstrap_run(
        tmp_path,
        _make_spec(
            discovery_metrics=[MetricName.COMBINED_AUC_K, MetricName.PATCH_EFFECT_RECOVERY]
        ),
    )
    tool = _discover_tool(handle)
    out, ok = asyncio.run(
        tool.handler(
            {"task": "demo", "reward_metric": "combined_auc_k", "dry_run": True}
        )
    )
    assert ok, out
    assert "combined_auc_k" in out


def test_explicit_off_spec_metric_refused_in_multi(tmp_path: Path):
    handle = _bootstrap_run(
        tmp_path,
        _make_spec(
            discovery_metrics=[MetricName.COMBINED_AUC_K, MetricName.PATCH_EFFECT_RECOVERY]
        ),
    )
    tool = _discover_tool(handle)
    out, ok = asyncio.run(
        tool.handler(
            {"task": "demo", "reward_metric": "auroc", "dry_run": True}
        )
    )
    assert not ok
    assert "not in the stage's pre-registered metrics" in out
