"""Components-substrate baseline algorithm.

Working starting point for component-substrate discovery: ranks attention
heads by ``mean(|z_clean − z_corrupt|)`` at the prediction position.
Gradient-free, ~1 forward pass per side, strong baseline on contrastive
tasks like IOI.

Copy this file as ``algorithm_v1.py`` in your session dir and edit
``score(...)`` to try a different attribution method. The harness will
import it and call ``score(pairs, loader=..., layers=..., top_k=...,
context=...)`` once per evaluation.
"""

from __future__ import annotations

from typing import Any

from autointerp.pipelines.investigation.discovery.candidate import Candidate

NAME = "components_baseline_mean_z_diff"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list[Candidate]:
    """Rank attention heads by ``mean(|z_clean − z_corrupt|)``.

    For each (layer, head) site, caches the per-head ``z`` activations on
    the clean and corrupted prompts and takes the mean absolute
    difference at the prediction position. Heads whose pre-W_O output
    differs the most between clean and corrupt are ranked highest.
    """
    import torch
    from autointerp.tools.head_patching import cache_head_z

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    clean_cache = cache_head_z(loader, clean_prompts, layers=layers)
    corrupt_cache = cache_head_z(loader, corrupt_prompts, layers=layers)

    # Each tensor is [B, S, H, d_head]. We score per head at the
    # prediction position (last token). Mean over batch, sum over d_head
    # of |Δ|.
    candidates: list[Candidate] = []
    for layer in layers:
        if layer not in clean_cache or layer not in corrupt_cache:
            continue
        clean_z = clean_cache[layer]   # [B, S, H, d_head]
        corrupt_z = corrupt_cache[layer]
        if clean_z.shape != corrupt_z.shape:
            # Token-length mismatch — should not happen with the IOI
            # length constraint but guard in case the benchmark relaxes.
            continue
        delta = (clean_z[:, -1, :, :] - corrupt_z[:, -1, :, :]).abs()  # [B, H, d_head]
        per_head = delta.mean(dim=0).sum(dim=-1)  # [H]
        for h in range(per_head.shape[0]):
            candidates.append(
                Candidate(
                    layer=int(layer),
                    idx=int(h),
                    score=float(per_head[h].item()),
                    kind="attn_head",
                    token_pos=-1,
                    metadata={"method": "mean_z_diff_at_-1"},
                )
            )

    candidates.sort(key=lambda c: -abs(c.score))
    return candidates[:top_k]
