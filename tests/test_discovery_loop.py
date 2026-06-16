"""Tests for the minimal hill-climbing discovery sub-agent.

Covers (a) the AUC-K metric impls and (b) the end-to-end dry-run loop, which
needs no model, GPU, or LLM credentials.
"""

from __future__ import annotations

import pytest

from autointerp.pipelines.investigation.discovery import run_discovery_subagent
from autointerp.pipelines.investigation.metrics import REGISTRY, MetricRegistryError
from autointerp.spec import METRIC_META, MetricName

# ---- AUC-K metrics ---------------------------------------------------------


def _compute(name: MetricName, inputs: dict) -> float:
    return REGISTRY[name].compute(inputs)


def test_mean_auc_k_perfect_curve() -> None:
    # A curve that is 1.0 everywhere integrates to 1.0.
    out = _compute(
        MetricName.MEAN_ABLATION_AUC_K,
        {"delta_curve_per_pair": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]},
    )
    assert out == pytest.approx(1.0)


def test_mean_auc_k_linear_ramp() -> None:
    # A 0->1 linear ramp integrates to 0.5 (triangle area / span).
    out = _compute(
        MetricName.MEAN_STEERING_AUC_K,
        {"delta_curve_per_pair": [[0.0, 0.5, 1.0]]},
    )
    assert out == pytest.approx(0.5)


def test_mean_auc_k_rejects_out_of_range() -> None:
    with pytest.raises(MetricRegistryError):
        _compute(
            MetricName.MEAN_ABLATION_AUC_K,
            {"delta_curve_per_pair": [[0.0, 1.5]]},
        )


def test_combined_auc_k_from_scalars() -> None:
    out = _compute(
        MetricName.COMBINED_AUC_K,
        {"mean_ablation_auc_k": 0.8, "mean_steering_auc_k": 0.6},
    )
    assert out == pytest.approx(0.7)


def test_combined_auc_k_from_curves() -> None:
    out = _compute(
        MetricName.COMBINED_AUC_K,
        {
            "ablation_delta_curve_per_pair": [[1.0, 1.0]],
            "steering_delta_curve_per_pair": [[0.0, 1.0]],
        },
    )
    assert out == pytest.approx(0.75)  # 0.5 * (1.0 + 0.5)


def test_auc_metrics_registered_and_metaed() -> None:
    for name in (
        MetricName.MEAN_ABLATION_AUC_K,
        MetricName.MEAN_STEERING_AUC_K,
        MetricName.COMBINED_AUC_K,
    ):
        assert name in REGISTRY
        assert name in METRIC_META
        assert METRIC_META[name].value_range == (0.0, 1.0)


# ---- dry-run loop ----------------------------------------------------------


def test_dry_run_loop_produces_best_candidate(tmp_path) -> None:
    result = run_discovery_subagent(
        session_dir=tmp_path / "disco",
        task="rank attention heads for the dry-run behavior",
        dry_run=True,
    )
    assert result.best_candidate == "algorithm_v1"
    assert result.best_reward is not None and 0.0 <= result.best_reward <= 1.0
    assert result.iterations_run == 1
    assert result.best_summary is not None
    assert result.best_summary["objective_metric"] == "combined_auc_k"
    # Scaffolded files exist.
    assert (result.session_dir / "algorithm_v1.py").exists()
    assert (result.session_dir / "results" / "algorithm_v1.json").exists()
    assert (result.session_dir / "loop_log.jsonl").exists()


def test_dry_run_loop_top_features_shape(tmp_path) -> None:
    result = run_discovery_subagent(
        session_dir=tmp_path / "disco",
        task="rank heads",
        top_k=3,
        dry_run=True,
    )
    feats = result.best_summary["top_features"]
    assert 1 <= len(feats) <= 3
    for f in feats:
        assert {"layer", "idx", "kind", "score"} <= set(f)
