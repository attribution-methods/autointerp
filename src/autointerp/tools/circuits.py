"""Circuit ranking and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class CircuitSite:
    name: str
    layer: int
    component: str
    score: float
    evidence: str = ""


def rank_sites(sites: Iterable[CircuitSite], top_k: Optional[int] = None) -> List[CircuitSite]:
    rows = sorted(sites, key=lambda site: abs(site.score), reverse=True)
    return rows[:top_k] if top_k is not None else rows


def combine_scores(
    *score_maps: Dict[str, float],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    if weights is None:
        weights = [1.0] * len(score_maps)
    combined: Dict[str, float] = {}
    for scores, weight in zip(score_maps, weights):
        for site, score in scores.items():
            combined[site] = combined.get(site, 0.0) + weight * score
    return combined


def formula_score(
    detection_delta: float = 0.0,
    identification_delta: float = 0.0,
    faithfulness_delta: float = 0.0,
    sparsity_penalty: float = 0.0,
) -> float:
    return detection_delta + identification_delta + faithfulness_delta - sparsity_penalty


def summarize_circuit(sites: Iterable[CircuitSite]) -> str:
    lines = []
    for site in rank_sites(sites):
        lines.append(
            f"{site.name}: L{site.layer} {site.component} "
            f"score={site.score:.4f} {site.evidence}".strip()
        )
    return "\n".join(lines)
