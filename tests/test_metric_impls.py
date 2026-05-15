"""Tests for the canonical metric impls added in this round.

The four metrics: AUROC, HIT_RATE, NECESSITY_DROP, COMPLETENESS, SUFFICIENCY.
"""

from __future__ import annotations

import pytest

from autointerp.pipelines.investigation.metrics import (
    REGISTRY,
    MetricRegistryError,
)
from autointerp.spec import MetricName


# ---- AUROC -----------------------------------------------------------------


def _auroc(inputs: dict) -> float:
    return REGISTRY[MetricName.AUROC].compute(inputs)


def test_auroc_perfect_separation() -> None:
    out = _auroc({"scores": [0.1, 0.2, 0.8, 0.9], "labels": [0, 0, 1, 1]})
    assert out == 1.0


def test_auroc_chance_with_ties() -> None:
    out = _auroc({"scores": [0.5, 0.5, 0.5, 0.5], "labels": [0, 0, 1, 1]})
    assert out == 0.5


def test_auroc_inverted_separation() -> None:
    out = _auroc({"scores": [0.1, 0.2, 0.8, 0.9], "labels": [1, 1, 0, 0]})
    assert out == 0.0


def test_auroc_realistic_value() -> None:
    # Sorted: 0.1(N), 0.4(N), 0.6(P), 0.65(N), 0.75(P), 0.8(P).
    # 8 of 9 score-pairs rank P > N (only 0.6 < 0.65 inverts) → AUC = 8/9.
    out = _auroc(
        {
            "scores": [0.10, 0.40, 0.65, 0.80, 0.60, 0.75],
            "labels": [0, 0, 0, 1, 1, 1],
        }
    )
    assert abs(out - 8 / 9) < 1e-9


def test_auroc_handles_booleans() -> None:
    out = _auroc({"scores": [0.1, 0.9], "labels": [False, True]})
    assert out == 1.0


def test_auroc_rejects_single_class() -> None:
    with pytest.raises(MetricRegistryError, match="positive and one negative"):
        _auroc({"scores": [0.1, 0.9], "labels": [1, 1]})


def test_auroc_rejects_non_binary_labels() -> None:
    with pytest.raises(MetricRegistryError, match="not 0/1"):
        _auroc({"scores": [0.1, 0.9], "labels": [0, 2]})


def test_auroc_length_mismatch() -> None:
    with pytest.raises(MetricRegistryError):
        _auroc({"scores": [0.1, 0.9, 0.5], "labels": [0, 1]})


# ---- HIT_RATE --------------------------------------------------------------


def _hit_rate(inputs: dict) -> float:
    return REGISTRY[MetricName.HIT_RATE].compute(inputs)


def test_hit_rate_basic() -> None:
    assert _hit_rate({"hits": [1, 1, 0, 0]}) == 0.5
    assert _hit_rate({"hits": [True, True, False]}) == pytest.approx(2 / 3)


def test_hit_rate_all_zero_or_one() -> None:
    assert _hit_rate({"hits": [0, 0, 0]}) == 0.0
    assert _hit_rate({"hits": [1, 1]}) == 1.0


def test_hit_rate_rejects_non_boolean() -> None:
    with pytest.raises(MetricRegistryError, match="not 0/1"):
        _hit_rate({"hits": [1, 0, 2]})


def test_hit_rate_rejects_empty() -> None:
    with pytest.raises(MetricRegistryError, match="non-empty"):
        _hit_rate({"hits": []})


# ---- NECESSITY_DROP --------------------------------------------------------


def _necessity_drop(inputs: dict) -> float:
    return REGISTRY[MetricName.NECESSITY_DROP].compute(inputs)


def test_necessity_drop_positive() -> None:
    out = _necessity_drop({"clean_metric": 0.9, "with_component_removed": 0.4})
    assert out == pytest.approx(0.5)


def test_necessity_drop_negative_means_redundant() -> None:
    # Removing the component made the behavior stronger → negative drop.
    out = _necessity_drop({"clean_metric": 0.5, "with_component_removed": 0.6})
    assert out == pytest.approx(-0.1)


# ---- COMPLETENESS ----------------------------------------------------------


def _completeness(inputs: dict) -> float:
    return REGISTRY[MetricName.COMPLETENESS].compute(inputs)


def test_completeness_perfect() -> None:
    assert _completeness({"full_model_metric": 0.8, "circuit_only_metric": 0.8}) == 1.0


def test_completeness_partial() -> None:
    out = _completeness({"full_model_metric": 1.0, "circuit_only_metric": 0.6})
    assert out == pytest.approx(0.6)


def test_completeness_rejects_zero_full() -> None:
    with pytest.raises(MetricRegistryError, match="undefined"):
        _completeness({"full_model_metric": 0.0, "circuit_only_metric": 0.5})


# ---- SUFFICIENCY -----------------------------------------------------------


def _sufficiency(inputs: dict) -> float:
    return REGISTRY[MetricName.SUFFICIENCY].compute(inputs)


def test_sufficiency_basic() -> None:
    assert _sufficiency({"full_model_metric": 1.0, "only_component_metric": 0.4}) == pytest.approx(0.4)


def test_sufficiency_rejects_zero_full() -> None:
    with pytest.raises(MetricRegistryError, match="undefined"):
        _sufficiency({"full_model_metric": 0.0, "only_component_metric": 0.5})


# ---- Registry coverage -----------------------------------------------------


def test_all_added_metrics_in_registry() -> None:
    # Quick guard so we don't accidentally drop one in a future refactor.
    for m in (
        MetricName.AUROC,
        MetricName.HIT_RATE,
        MetricName.NECESSITY_DROP,
        MetricName.COMPLETENESS,
        MetricName.SUFFICIENCY,
    ):
        assert m in REGISTRY
