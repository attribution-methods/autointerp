"""MLP-neuron evaluator (stub).

The structure is identical to ``attn_heads.evaluate``: load the model
once, cache MLP-layer activations, run the candidate's ``score(...)`` to
get ranked ``Candidate(kind="mlp_neuron")`` rows, ablate top-K neurons
per layer, integrate the AUC-K curve, return the harness payload.

What's missing for a real implementation:

1. A neuron-ablation primitive analogous to
   ``autointerp.tools.head_patching.mean_ablate_heads``. The primitive
   should hook into the MLP's down-projection input (i.e. the post-
   activation pre-W_out vector, shape ``[batch, seq, d_mlp]``) and zero
   or mean-ablate selected neuron indices. Add it under
   ``autointerp.tools.mlp_patching`` (new module).
2. A ``cache_mlp_activations`` helper to build the mean-ablation
   baseline (same shape: ``[batch, seq, d_mlp]`` per layer).
3. Adapt the K-sweep loop from ``attn_heads.evaluate`` to use the new
   primitive — most of the file copies over verbatim.

Adding this evaluator should NOT require changes to the harness, the
loop, the registry, or any spec — only this file plus the new
``mlp_patching`` primitives.
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
        "mlp_neurons evaluator is not yet implemented. To wire it: (1) add "
        "`mean_ablate_neurons` and `cache_mlp_activations` to "
        "autointerp.tools.mlp_patching; (2) port the K-sweep loop from "
        "evaluators/components/attn_heads.py:evaluate. The spec layer "
        "already supports component_kinds=['mlp_neuron']."
    )


__all__ = ["evaluate"]
