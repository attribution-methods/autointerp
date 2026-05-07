"""Layer-normalized activation difference scoring.

Extends the baseline mean |z_clean - z_corrupt| approach by normalizing
scores within each layer. This corrects for varying activation scales across
layers and prevents late layers from dominating purely due to magnitude.

For each head, computes the standardized score relative to other heads in
the same layer, making cross-layer comparisons more meaningful.
"""

from __future__ import annotations

from typing import Any

NAME = "layer_normalized_z_diff"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list:
    """Rank heads by layer-normalized activation differences.

    Like algorithm_v1, but applies layer-wise z-score normalization to make
    scores comparable across layers. Within each layer, computes:
    score_norm = (score_raw - mean) / std

    This prevents layer-specific activation scales from biasing the ranking.
    """
    import torch
    from autointerp.tools.head_patching import cache_head_z
    from autointerp.pipelines.investigation.discovery.candidate import Candidate

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    clean_cache = cache_head_z(loader, clean_prompts, layers=layers)
    corrupt_cache = cache_head_z(loader, corrupt_prompts, layers=layers)

    # First pass: compute raw scores per layer
    raw_scores_by_layer = {}  # layer -> [scores per head]

    for layer in layers:
        if layer not in clean_cache or layer not in corrupt_cache:
            continue
        clean_z = clean_cache[layer]
        corrupt_z = corrupt_cache[layer]
        if clean_z.shape != corrupt_z.shape:
            continue

        # Mean absolute difference at prediction position
        delta = (clean_z[:, -1, :, :] - corrupt_z[:, -1, :, :]).abs()  # [B, H, d_head]
        per_head = delta.mean(dim=0).sum(dim=-1)  # [H]
        raw_scores_by_layer[layer] = per_head

    # Second pass: normalize within each layer and create candidates
    candidates: list[Candidate] = []

    for layer in layers:
        if layer not in raw_scores_by_layer:
            continue

        scores = raw_scores_by_layer[layer]  # [H]

        # Z-score normalization within layer
        mean_score = scores.mean()
        std_score = scores.std()

        # Avoid division by zero
        if std_score < 1e-8:
            normalized_scores = scores - mean_score
        else:
            normalized_scores = (scores - mean_score) / std_score

        for h in range(scores.shape[0]):
            candidates.append(
                Candidate(
                    layer=int(layer),
                    idx=int(h),
                    score=float(normalized_scores[h].item()),
                    kind="attn_head",
                    token_pos=-1,
                    metadata={
                        "method": "layer_normalized_z_diff",
                        "raw_score": float(scores[h].item()),
                    },
                )
            )

    # Sort by absolute normalized score
    candidates.sort(key=lambda c: -abs(c.score))
    return candidates[:top_k]
