"""Tests for custom metrics: spec-time validation and runtime dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import (
    MetricRegistryError,
    compute_metric,
    init_run,
)
from autointerp.schemas import BehaviorSpec
from autointerp.spec import (
    Approval,
    Budget,
    ContrastSpec,
    Criterion,
    CustomMetricDef,
    DatasetSpec,
    InvestigationSpec,
    InvestigationStage,
    MetricFamily,
    MetricName,
    ModelRef,
    PatternId,
    SpecStatus,
    StageSpec,
    ToolName,
)


# ---- helpers ---------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


_ENTROPY_SOURCE = (
    "def compute(inputs):\n"
    "    weights = inputs['attention_weights']\n"
    "    total = 0.0\n"
    "    n = 0\n"
    "    for row in weights:\n"
    "        s = 0.0\n"
    "        for p in row:\n"
    "            if p > 0:\n"
    "                s -= p * math.log(p)\n"
    "        total += s\n"
    "        n += 1\n"
    "    return total / n\n"
)


def _entropy_def() -> CustomMetricDef:
    return CustomMetricDef(
        name="attention_entropy",
        description="Mean Shannon entropy of attention rows.",
        family=MetricFamily.BEHAVIORAL,
        value_range=(0.0, None),
        direction="lower",
        requires_inputs=["attention_weights"],
        source_code=_ENTROPY_SOURCE,
    )


def _spec_with_custom(
    defn: CustomMetricDef | None = None,
    *,
    threshold: float = 2.5,
    comparator: str = "<=",
) -> InvestigationSpec:
    defn = defn or _entropy_def()
    return InvestigationSpec(
        spec_id="custom-metric-fixture",
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
                metrics=[MetricName.CUSTOM],
            )
        ],
        success_criteria=[
            Criterion(
                criterion_id="c-entropy",
                description="entropy threshold",
                metric=MetricName.CUSTOM,
                comparator=comparator,
                threshold=threshold,
                on_split="dev",
                custom_metric_def=defn,
            )
        ],
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=_now()),
    )


# ---- spec-time validation --------------------------------------------------


def test_custom_metric_def_hashes_source() -> None:
    defn = _entropy_def()
    expected = hashlib.sha256(_ENTROPY_SOURCE.encode("utf-8")).hexdigest()
    assert defn.source_hash == expected


def test_custom_metric_def_rejects_forbidden_token() -> None:
    # Non-import forbidden token (the import check would otherwise fire first).
    bad = "def compute(inputs):\n    return float(eval('1+1'))\n"
    with pytest.raises(ValueError, match="forbidden token"):
        CustomMetricDef(
            name="bad", description="desc", family=MetricFamily.BEHAVIORAL,
            value_range=(0.0, 1.0), direction="higher",
            requires_inputs=["x"], source_code=bad,
        )


def test_custom_metric_def_rejects_missing_function() -> None:
    src = "def something_else(x):\n    return 0.0\n"
    with pytest.raises(ValueError, match="must define def compute"):
        CustomMetricDef(
            name="bad2", description="desc", family=MetricFamily.BEHAVIORAL,
            value_range=(0.0, 1.0), direction="higher",
            requires_inputs=["x"], source_code=src,
        )


def test_custom_metric_def_rejects_syntax_error() -> None:
    src = "def compute(::\n    return 0.0\n"
    with pytest.raises(ValueError, match="does not parse"):
        CustomMetricDef(
            name="bad3", description="desc", family=MetricFamily.BEHAVIORAL,
            value_range=(0.0, 1.0), direction="higher",
            requires_inputs=["x"], source_code=src,
        )


def test_criterion_with_custom_metric_validates() -> None:
    spec = _spec_with_custom()
    assert spec.success_criteria[0].custom_metric_def is not None
    assert spec.success_criteria[0].custom_metric_def.name == "attention_entropy"


def test_criterion_custom_without_def_rejected() -> None:
    with pytest.raises(ValueError, match="custom_metric_def"):
        Criterion(
            criterion_id="x", description="desc", metric=MetricName.CUSTOM,
            comparator=">=", threshold=0.5, on_split="dev",
        )


def test_criterion_non_custom_with_def_rejected() -> None:
    defn = _entropy_def()
    with pytest.raises(ValueError, match="only allowed when metric=CUSTOM"):
        Criterion(
            criterion_id="x", description="desc", metric=MetricName.ACCURACY,
            comparator=">=", threshold=0.5, on_split="dev",
            custom_metric_def=defn,
        )


def test_custom_threshold_outside_range_rejected() -> None:
    defn = _entropy_def()  # range (0.0, None), direction lower
    with pytest.raises(ValueError, match="below custom metric"):
        Criterion(
            criterion_id="x", description="desc", metric=MetricName.CUSTOM,
            comparator="<=", threshold=-0.1, on_split="dev",
            custom_metric_def=defn,
        )


def test_custom_comparator_direction_mismatch_rejected() -> None:
    defn = _entropy_def()  # direction lower
    with pytest.raises(ValueError, match="higher-is-better"):
        Criterion(
            criterion_id="x", description="desc", metric=MetricName.CUSTOM,
            comparator=">=", threshold=2.0, on_split="dev",
            custom_metric_def=defn,
        )


# ---- runtime dispatch ------------------------------------------------------


def test_compute_metric_dispatches_to_custom(tmp_path: Path) -> None:
    spec = _spec_with_custom()
    handle = init_run(spec, runs_root=tmp_path)
    # Two-row uniform attention over 2 positions -> entropy = log(2) per row.
    inputs = {
        "__custom_name__": "attention_entropy",
        "attention_weights": [[0.5, 0.5], [0.5, 0.5]],
    }
    payload, token = compute_metric(
        handle, metric="custom", metric_id="m1",
        inputs=inputs, threshold=2.5, comparator="<=",
    )
    import math
    assert math.isclose(payload["value"], math.log(2), rel_tol=1e-9)
    assert payload["passed"] is True
    md = payload["metadata"]
    assert md["custom"] is True
    assert md["metric_name"] == "attention_entropy"
    assert md["source_hash"] == spec.success_criteria[0].custom_metric_def.source_hash
    assert token.startswith("tok_")


def test_compute_metric_unknown_custom_name_rejected(tmp_path: Path) -> None:
    spec = _spec_with_custom()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(MetricRegistryError, match="unknown custom metric"):
        compute_metric(
            handle, metric="custom", metric_id="m1",
            inputs={"__custom_name__": "not_in_spec", "attention_weights": [[1.0]]},
        )


def test_compute_metric_custom_missing_inputs_rejected(tmp_path: Path) -> None:
    spec = _spec_with_custom()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(MetricRegistryError, match="missing required inputs"):
        compute_metric(
            handle, metric="custom", metric_id="m1",
            inputs={"__custom_name__": "attention_entropy"},
        )


def test_compute_metric_custom_no_lookup_key_rejected(tmp_path: Path) -> None:
    spec = _spec_with_custom()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(MetricRegistryError, match="__custom_name__"):
        compute_metric(
            handle, metric="custom", metric_id="m1",
            inputs={"attention_weights": [[1.0]]},
        )


def test_compute_metric_custom_clipped_to_range(tmp_path: Path) -> None:
    bounded = CustomMetricDef(
        name="bounded_score",
        description="returns min(inputs['x'], 1.0).",
        family=MetricFamily.BEHAVIORAL,
        value_range=(0.0, 1.0),
        direction="higher",
        requires_inputs=["x"],
        source_code="def compute(inputs):\n    return float(inputs['x'])\n",
    )
    spec = _spec_with_custom(bounded, threshold=0.5, comparator=">=")
    handle = init_run(spec, runs_root=tmp_path)
    payload, _ = compute_metric(
        handle, metric="custom", metric_id="m1",
        inputs={"__custom_name__": "bounded_score", "x": 5.0},
    )
    assert payload["value"] == 1.0
    assert payload["metadata"]["unclipped_value"] == 5.0


# ---- planner-facing tool ---------------------------------------------------


def test_propose_custom_metric_tool_validates_and_echoes() -> None:
    from autointerp_agent.stage0_tools import _propose_custom_metric

    args = {
        "name": "attention_entropy",
        "description": "Mean Shannon entropy of attention rows.",
        "family": "behavioral",
        "value_range": [0.0, None],
        "direction": "lower",
        "requires_inputs": ["attention_weights"],
        "source_code": _ENTROPY_SOURCE,
    }
    msg, ok = asyncio.run(_propose_custom_metric(args))
    assert ok
    assert "Validated custom metric 'attention_entropy'" in msg
    assert "sha256=" in msg


def test_propose_custom_metric_tool_rejects_imports() -> None:
    from autointerp_agent.stage0_tools import _propose_custom_metric

    args = {
        "name": "bad",
        "description": "desc",
        "family": "behavioral",
        "value_range": [0.0, 1.0],
        "direction": "higher",
        "requires_inputs": ["x"],
        "source_code": "def compute(inputs):\n    import os\n    return 0.0\n",
    }
    msg, ok = asyncio.run(_propose_custom_metric(args))
    assert not ok
    assert "import statements are not allowed" in msg
