"""Probe training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeReport:
    probe_type: str
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    auroc_mean: float
    auroc_std: float
    n_samples: int
    n_positive: int
    n_negative: int


def train_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    probe_type: str = "linear",
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Tuple[Any, StandardScaler, ProbeReport]:
    scaler = StandardScaler()
    x = scaler.fit_transform(activations)
    y = labels.astype(int)
    if len(set(y.tolist())) != 2:
        raise ValueError("Probe labels must contain exactly two classes")

    if probe_type == "linear":
        probe = LogisticRegression(max_iter=1000, random_state=random_seed)
    elif probe_type == "mlp":
        probe = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=random_seed)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    metrics: Dict[str, list] = {"accuracy": [], "f1": [], "auroc": []}
    folds = min(cv_folds, np.bincount(y).min())
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in splitter.split(x, y):
        fold_probe = probe.__class__(**probe.get_params())
        fold_probe.fit(x[train_idx], y[train_idx])
        pred = fold_probe.predict(x[test_idx])
        metrics["accuracy"].append(accuracy_score(y[test_idx], pred))
        metrics["f1"].append(f1_score(y[test_idx], pred))
        if hasattr(fold_probe, "predict_proba"):
            prob = fold_probe.predict_proba(x[test_idx])[:, 1]
            metrics["auroc"].append(roc_auc_score(y[test_idx], prob))

    probe.fit(x, y)
    report = ProbeReport(
        probe_type=probe_type,
        accuracy_mean=float(np.mean(metrics["accuracy"])),
        accuracy_std=float(np.std(metrics["accuracy"])),
        f1_mean=float(np.mean(metrics["f1"])),
        f1_std=float(np.std(metrics["f1"])),
        auroc_mean=float(np.mean(metrics["auroc"])) if metrics["auroc"] else float("nan"),
        auroc_std=float(np.std(metrics["auroc"])) if metrics["auroc"] else float("nan"),
        n_samples=int(len(y)),
        n_positive=int(np.sum(y == 1)),
        n_negative=int(np.sum(y == 0)),
    )
    return probe, scaler, report


def probe_direction(probe: Any, scaler: StandardScaler) -> np.ndarray:
    if not hasattr(probe, "coef_"):
        raise ValueError("Only linear probes expose a single direction")
    return probe.coef_[0] / scaler.scale_
