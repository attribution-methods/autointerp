"""SAE-feature evaluator (stub).

Wire one of these per SAE family:

- ``sae_gemmascope`` — Gemma-2 + Google's GemmaScope SAEs (the reference
  case from ``circuitbreaker/auto_circuit_discovery``).
- ``sae_bloom_gpt2_small`` — Joseph Bloom's GPT-2-small SAEs (HF).
- ``sae_pythia`` — EleutherAI / TransformerLens SAEs.

The reference repo's ``auto_circuit_discovery/harness.py`` already
implements the Gemma+GemmaScope path; porting it here means:

1. Loading ``model + saes`` once via the appropriate HF / sae_lens path
   (a new helper in ``autointerp.tools.sae`` or
   ``autointerp.tools.transcoder_loader``). Cache like
   ``components.attn_heads._MODEL_CACHE``.
2. Building the harness payload: per K, project the residual stream onto
   the top-K SAE features (clean direction); steer or ablate; measure
   normalized delta; integrate AUC-K.
3. Returning the same dict shape ``components.attn_heads.evaluate``
   returns (``mean_ablation_auc_k``, ``mean_steering_auc_k``,
   ``combined_auc_k``, ``top_features``, ``per_pair``).

The spec layer + registry + harness routing already work for SAE — only
this file needs filling in for any given SAE family.
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
    decomposition = (context or {}).get("decomposition", "<unset>")
    raise NotImplementedError(
        f"SAE-feature evaluator (decomposition={decomposition!r}) is not "
        "yet implemented. Port the relevant SAE path — for "
        "sae_gemmascope, see "
        "https://github.com/attribution-methods/circuitbreaker/blob/auto-circuit-discovery/auto_circuit_discovery/harness.py"
    )


__all__ = ["evaluate"]
