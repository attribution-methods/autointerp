"""Path-patching attribution algorithm.

Ranks attention heads by their *direct* effect on the logit_diff metric using
path patching. Unlike head_patch_sweep which allows downstream heads to adapt,
path patching isolates the causal contribution of each head by freezing all
other heads to their corrupt values and measuring only the direct effect on
the final logits.

This provides more precise causal attribution by eliminating indirect effects
through later layers.
"""

from __future__ import annotations

from typing import Any

from autointerp.pipelines.investigation.discovery.candidate import Candidate

NAME = "path_patch_attribution"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list[Candidate]:
    """Rank attention heads by direct path-patching effect on logit_diff.

    For each (layer, head) site, performs path patching: patches that head's
    clean z activation into the corrupt run while keeping all other activations
    at their corrupt values. This isolates the direct causal effect of each
    head on the final logits, without indirect effects through downstream heads.

    More precise than head_patch_sweep because it measures direct rather than
    total (direct + indirect) effects.
    """
    import torch
    from autointerp.tools.head_patching import (
        cache_head_z,
        logit_diff,
        run_with_head_patches,
        HeadPatch,
    )

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    # Get target/foil token IDs for logit_diff computation
    target_ids = [p.target_token_id for p in pairs]
    foil_ids = [p.foil_token_id for p in pairs]

    # Cache activations for both clean and corrupt runs
    clean_cache = cache_head_z(loader, clean_prompts, layers=layers)
    corrupt_cache = cache_head_z(loader, corrupt_prompts, layers=layers)

    # Compute baseline logit_diffs for normalization
    with torch.no_grad():
        clean_inputs = loader.tokenizer(clean_prompts, return_tensors="pt", padding=True)
        corrupt_inputs = loader.tokenizer(corrupt_prompts, return_tensors="pt", padding=True)

        clean_logits = loader.model(clean_inputs.input_ids.to(loader.device)).logits
        corrupt_logits = loader.model(corrupt_inputs.input_ids.to(loader.device)).logits

        clean_ld = logit_diff(clean_logits, target_ids, foil_ids)
        corrupt_ld = logit_diff(corrupt_logits, target_ids, foil_ids)
        gap = clean_ld - corrupt_ld

    candidates: list[Candidate] = []

    # Path patching: for each head, patch clean z into corrupt while keeping
    # all other heads at corrupt values
    for layer in layers:
        if layer not in clean_cache or layer not in corrupt_cache:
            continue

        n_heads = clean_cache[layer].shape[2]  # [B, S, H, d_head]

        for head_idx in range(n_heads):
            # Create a patch that overrides only this specific (layer, head) at position -1
            # All other heads remain at their corrupt values
            # source needs to be [B, 1, d_head] or broadcastable
            patch_value = clean_cache[layer][:, -1, head_idx, :]  # [B, d_head]

            patches = [
                HeadPatch(
                    layer=layer,
                    head=head_idx,
                    source=patch_value,
                    positions=[-1],
                )
            ]

            # Run corrupt forward pass with this single head patched from clean
            patched_logits = run_with_head_patches(
                loader,
                corrupt_prompts,
                patches=patches,
            )

            patched_ld = logit_diff(patched_logits, target_ids, foil_ids)

            # Compute patch_effect_recovery: (patched - corrupt) / (clean - corrupt)
            if abs(gap) > 1e-6:
                recovery = (patched_ld - corrupt_ld) / gap
            else:
                recovery = torch.tensor(0.0)

            candidates.append(
                Candidate(
                    layer=int(layer),
                    idx=int(head_idx),
                    score=float(recovery.item()),
                    kind="attn_head",
                    token_pos=-1,
                    metadata={
                        "method": "path_patch",
                        "patch_position": -1,
                        "direct_effect": float((patched_ld - corrupt_ld).item()),
                    },
                )
            )

    # Sort by absolute recovery (both positive and negative effects matter)
    candidates.sort(key=lambda c: -abs(c.score))
    return candidates[:top_k]
