"""Sparse autoencoder feature utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch


@dataclass
class SAEFeature:
    feature_id: int
    activation: float
    label: str = ""
    position: Optional[int] = None
    token: str = ""


def top_features(feature_acts: torch.Tensor, top_k: int = 50) -> List[SAEFeature]:
    flat = feature_acts.detach().cpu().flatten()
    values, ids = torch.topk(flat, k=min(top_k, flat.numel()))
    return [
        SAEFeature(feature_id=int(idx), activation=float(value))
        for value, idx in zip(values, ids)
    ]


def attach_labels(features: Iterable[SAEFeature], labels: Dict[int, str]) -> List[SAEFeature]:
    rows = []
    for feature in features:
        rows.append(
            SAEFeature(
                feature_id=feature.feature_id,
                activation=feature.activation,
                label=labels.get(feature.feature_id, feature.label),
                position=feature.position,
                token=feature.token,
            )
        )
    return rows


def filter_semantic_features(
    features: Iterable[SAEFeature],
    banned_terms: Optional[List[str]] = None,
) -> List[SAEFeature]:
    banned = banned_terms or ["punctuation", "whitespace", "format", "syntax", "newline"]
    kept = []
    for feature in features:
        label = feature.label.lower()
        if label and any(term in label for term in banned):
            continue
        kept.append(feature)
    return kept


def format_feature_cluster(features: Iterable[SAEFeature]) -> str:
    lines = []
    for feature in features:
        label = f" - {feature.label}" if feature.label else ""
        token = f" token={feature.token!r}" if feature.token else ""
        lines.append(f"{feature.feature_id}: activation={feature.activation:.4f}{token}{label}")
    return "\n".join(lines)
