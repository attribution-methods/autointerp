"""Head-patch-sweep ranking algorithm.

Directly ranks attention heads by their patch_effect_recovery measured via
head_patch_sweep. This is a direct optimization of the target metric: for
each (layer, head) pair, we patch clean head z activations into the corrupt
run and measure the fraction of clean-corrupt logit_diff gap recovered.

This should align better with the optimization objective than algorithm_v1's
mean z-diff, which is only an indirect proxy for causal effect.
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
    """Rank attention heads by patch_effect_recovery from head_patch_sweep.

    For each (layer, head) site, patches clean head z activations into the
    corrupt run and measures the fraction of clean-corrupt logit_diff gap
    recovered at the prediction position. Heads with highest
    patch_effect_recovery are ranked first.
    """
    from autointerp.tools.head_patching import head_patch_sweep

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    # Extract IO and S token indices for logit_diff computation
    # Assuming the pairs have answer_token_idx and foil_token_idx attributes
    answer_tokens = [p.answer_token_idx for p in pairs]
    foil_tokens = [p.foil_token_idx for p in pairs]

    # Run head_patch_sweep: patches clean head z into corrupt run for all heads
    # Returns dict[layer][head] -> patch_effect_recovery
    sweep_results = head_patch_sweep(
        loader=loader,
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        answer_tokens=answer_tokens,
        foil_tokens=foil_tokens,
        layers=layers,
    )

    # Convert sweep results to Candidate list
    candidates: list[Candidate] = []
    for layer in layers:
        if layer not in sweep_results:
            continue
        per_head_effects = sweep_results[layer]  # dict or tensor indexed by head

        # Handle both dict and tensor results
        if hasattr(per_head_effects, 'shape'):
            # It's a tensor [H]
            for h in range(per_head_effects.shape[0]):
                candidates.append(
                    Candidate(
                        layer=int(layer),
                        idx=int(h),
                        score=float(per_head_effects[h].item()),
                        kind="attn_head",
                        token_pos=-1,
                        metadata={"method": "head_patch_sweep"},
                    )
                )
        else:
            # It's a dict {head: effect}
            for h, effect in per_head_effects.items():
                candidates.append(
                    Candidate(
                        layer=int(layer),
                        idx=int(h),
                        score=float(effect),
                        kind="attn_head",
                        token_pos=-1,
                        metadata={"method": "head_patch_sweep"},
                    )
                )

    # Sort by absolute score (some heads may inhibit the behavior, negative score)
    # But we want heads that promote the clean behavior (positive score) first
    candidates.sort(key=lambda c: -c.score)
    return candidates[:top_k]
