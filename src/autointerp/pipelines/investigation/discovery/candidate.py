"""The unit a discovery candidate algorithm returns.

A ``score(...)`` function ranks model components / features and returns a list
of ``Candidate`` objects sorted by descending ``abs(score)``. The harness reads
these to build the ablation / steering AUC curves that define the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A single ranked site (attention head, MLP neuron, SAE feature, ...)."""

    layer: int
    idx: int
    score: float
    kind: str = "attn_head"
    token_pos: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["Candidate"]
