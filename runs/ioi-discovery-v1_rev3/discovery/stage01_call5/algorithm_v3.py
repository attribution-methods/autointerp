"""Head-patch-sweep ranking algorithm.

Directly ranks attention heads by their patch_effect_recovery when patching
clean head z activations into corrupt runs. This aligns the scoring metric
with the optimization objective, measuring the causal effect of each head
on the logit_diff metric.

Unlike algorithm_v1's indirect mean(|z_clean - z_corrupt|) proxy, this
sweeps all (layer, head) pairs and measures their actual contribution to
recovering the clean behavior when patched into the corrupt run.
"""

from __future__ import annotations

from typing import Any

from autointerp.pipelines.investigation.discovery.candidate import Candidate

NAME = "head_patch_sweep_ranking"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list[Candidate]:
    """Rank attention heads by direct patch_effect_recovery measurement.

    For each (layer, head) site, runs a full clean->corrupt patching sweep
    and ranks heads by their patch_effect_recovery: the fraction of the
    clean-corrupt logit_diff gap recovered by patching that head's clean
    activation into the corrupt run.

    This is the most direct scoring method since patch_effect_recovery is
    exactly what we're optimizing for in the evaluation metric.
    """
    import torch
    from autointerp.tools.head_patching import head_patch_sweep, logit_diff

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    # Get target/foil token IDs for logit_diff computation
    # In IOI: target=IO (indirect object), foil=S (subject)
    target_ids = [p.target_token_id for p in pairs]
    foil_ids = [p.foil_token_id for p in pairs]

    # Create a metric function that computes logit_diff for this batch
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logit_diff(logits, target_ids, foil_ids)

    # Run the full sweep: for each (layer, head), patch clean z into corrupt
    # and measure the resulting patch_effect_recovery
    sweep_results = head_patch_sweep(
        handle=loader,
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        metric=metric,
        layers=layers,
        patch_positions=[-1],  # Only patch at prediction position
    )

    # sweep_results is a dict with "recovery": [L, H] tensor
    # and "layers", "heads" lists for indexing
    recovery = sweep_results["recovery"]  # [L, H]
    layer_list = sweep_results["layers"]
    head_list = sweep_results["heads"]

    candidates: list[Candidate] = []
    for li, layer in enumerate(layer_list):
        for hi, head in enumerate(head_list):
            per_value = float(recovery[li, hi].item())
            candidates.append(
                Candidate(
                    layer=int(layer),
                    idx=int(head),
                    score=per_value,
                    kind="attn_head",
                    token_pos=-1,
                    metadata={"method": "head_patch_sweep", "patch_position": -1},
                )
            )

    # Sort by absolute patch_effect_recovery (keep sign for metadata)
    # Positive = promotes clean behavior, negative = inhibits it
    # Both are causally important
    candidates.sort(key=lambda c: -abs(c.score))
    return candidates[:top_k]
