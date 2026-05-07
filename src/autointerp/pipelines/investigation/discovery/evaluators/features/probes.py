"""Probe-direction evaluator (stub). For learned probes / contrastive
directions / PCA components — anything where the "feature" is a single
residual-stream direction. Intervention: project the residual onto the
direction, scale, add back."""

from __future__ import annotations

from typing import Any, Callable

from autointerp.pipelines.investigation.discovery.candidate import Candidate


def evaluate(
    score_fn: Callable[..., list[Candidate]],
    *,
    model_id: str,
    device: str | None,
    pairs: list[Any],
    behavioral_metric: Callable[[Any, Any], float],
    layers: list[int] | None,
    top_k: int,
    k_grid: list[int],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError(
        "probe-direction evaluator is not yet implemented. Hook "
        "autointerp.tools.vectors and autointerp.tools.generation; "
        "build the dataclass payload as in components/attn_heads.py."
    )


__all__ = ["evaluate"]
