"""MLP-layer evaluator (stub). See ``mlp_neurons.py`` for the structure
to mirror — at layer granularity the intervention is whole-layer
zeroing or mean-ablation."""

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
        "mlp_layers evaluator is not yet implemented. The spec layer "
        "already supports component_kinds=['mlp_layer']; wire ablation "
        "primitives in autointerp.tools.mlp_patching and copy the "
        "K-sweep loop from evaluators/components/attn_heads.py."
    )


__all__ = ["evaluate"]
