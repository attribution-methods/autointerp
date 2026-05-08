"""Residual-layer site evaluator (stub).

Targets the residual stream at a specific (layer, token_pos). Useful for
phenomena where you want to localize "which layer's residual carries the
behavioral signal" without committing to attn vs MLP. Intervention:
replace the residual at that site with the mean over a baseline
distribution.
"""

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
        "residual_layers evaluator is not yet implemented. Add residual-"
        "stream patching to autointerp.tools.patching and adapt the "
        "K-sweep loop from evaluators/components/attn_heads.py."
    )


__all__ = ["evaluate"]
