"""Unit tests for the new AUC reward metrics."""

from __future__ import annotations

import math

import pytest

from autointerp.pipelines.investigation.metrics import (
    REGISTRY,
    MetricRegistryError,
    _auroc,
    _combined_auc_k,
    _mean_auc_k,
)
from autointerp.spec import MetricName


# ---- auroc -----------------------------------------------------------------


def test_auroc_perfect_separation():
    assert _auroc({"scores": [0.1, 0.2, 0.3, 0.4], "labels": [0, 0, 1, 1]}) == 1.0


def test_auroc_inverted_separation():
    assert _auroc({"scores": [0.1, 0.2, 0.3, 0.4], "labels": [1, 1, 0, 0]}) == 0.0


def test_auroc_random_ties():
    # All scores equal → mid-rank gives 0.5 regardless of label distribution.
    assert _auroc({"scores": [0.5] * 4, "labels": [1, 0, 1, 0]}) == 0.5


def test_auroc_single_class_rejected():
    with pytest.raises(MetricRegistryError, match="both classes"):
        _auroc({"scores": [0.1, 0.2, 0.3], "labels": [1, 1, 1]})


def test_auroc_label_value_rejected():
    with pytest.raises(MetricRegistryError, match=r"not in \{0, 1\}"):
        _auroc({"scores": [0.1, 0.2], "labels": [0, 2]})


def test_auroc_length_mismatch_rejected():
    with pytest.raises(MetricRegistryError, match="equal length"):
        _auroc({"scores": [0.1, 0.2, 0.3], "labels": [0, 1]})


def test_auroc_in_registry():
    assert MetricName.AUROC in REGISTRY
    impl = REGISTRY[MetricName.AUROC]
    assert impl.required_inputs == ("scores", "labels")


# ---- mean_*_auc_k ----------------------------------------------------------


def test_mean_auc_k_linear_curve():
    # [0, 0.5, 1] integrated by trapezoid over k_grid [0,1,2] = 1.0; normalized
    # by x-range (=2) → 0.5.
    res = _mean_auc_k(
        {"delta_curve_per_pair": [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]},
        name=MetricName.MEAN_ABLATION_AUC_K,
    )
    assert math.isclose(res, 0.5, abs_tol=1e-9)


def test_mean_auc_k_constant_one_curve():
    # Saturated curve [1, 1, 1] integrates to area 2 over span 2 → 1.0.
    res = _mean_auc_k(
        {"delta_curve_per_pair": [[1.0, 1.0, 1.0]]},
        name=MetricName.MEAN_ABLATION_AUC_K,
    )
    assert math.isclose(res, 1.0, abs_tol=1e-9)


def test_mean_auc_k_explicit_grid():
    # Non-uniform grid: trapezoid over [0,1,4] with curve [0, 0.5, 1].
    # Area = 0.5*1*0.5 + 0.5*3*1.5 = 0.25 + 2.25 = 2.5; span=4 → 0.625.
    res = _mean_auc_k(
        {
            "delta_curve_per_pair": [[0.0, 0.5, 1.0]],
            "k_grid": [0.0, 1.0, 4.0],
        },
        name=MetricName.MEAN_ABLATION_AUC_K,
    )
    assert math.isclose(res, 0.625, abs_tol=1e-9)


def test_mean_auc_k_out_of_range_rejected():
    with pytest.raises(MetricRegistryError, match=r"not in \[0, 1\]"):
        _mean_auc_k(
            {"delta_curve_per_pair": [[0.0, 1.5]]},
            name=MetricName.MEAN_ABLATION_AUC_K,
        )


def test_mean_auc_k_short_curve_rejected():
    with pytest.raises(MetricRegistryError, match="length >= 2"):
        _mean_auc_k(
            {"delta_curve_per_pair": [[0.5]]},
            name=MetricName.MEAN_ABLATION_AUC_K,
        )


# ---- combined_auc_k --------------------------------------------------------


def test_combined_direct_scalars():
    assert _combined_auc_k(
        {"mean_ablation_auc_k": 0.4, "mean_steering_auc_k": 0.6}
    ) == 0.5


def test_combined_from_curves():
    res = _combined_auc_k(
        {
            "ablation_delta_curve_per_pair": [[0.0, 1.0, 1.0]],
            "steering_delta_curve_per_pair": [[0.0, 0.0, 1.0]],
        }
    )
    # ablation: trapezoid of [0,1,1] / 2 = (0.5 + 1.0)/2 = 0.75
    # steering: trapezoid of [0,0,1] / 2 = (0 + 0.5)/2 = 0.25
    # combined = 0.5
    assert math.isclose(res, 0.5, abs_tol=1e-9)


def test_combined_in_registry():
    assert MetricName.COMBINED_AUC_K in REGISTRY
    assert MetricName.MEAN_ABLATION_AUC_K in REGISTRY
    assert MetricName.MEAN_STEERING_AUC_K in REGISTRY
