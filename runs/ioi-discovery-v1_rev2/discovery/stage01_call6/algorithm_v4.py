"""Position-aggregated z-diff algorithm.

Instead of only looking at the prediction position, aggregate activation
differences across multiple positions near the end of the sequence. In IOI,
critical computation happens across several tokens (duplicate name detection,
indirect object retrieval, prediction). Summing |z_clean - z_corrupt| across
the last few positions captures heads active throughout this computation.
"""

from __future__ import annotations

from typing import Any

from autointerp.pipelines.investigation.discovery.candidate import Candidate

NAME = "position_aggregated_z_diff"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list[Candidate]:
    """Rank attention heads by aggregated ``mean(|z_clean − z_corrupt|)`` across positions.

    For each (layer, head) site, caches the per-head ``z`` activations on
    the clean and corrupted prompts and computes mean absolute difference
    across the last N positions (not just the final token). This captures
    heads that contribute to IOI computation across multiple critical
    positions (duplicate name detection, retrieval, prediction).
    """
    import torch
    from autointerp.tools.head_patching import cache_head_z

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    clean_cache = cache_head_z(loader, clean_prompts, layers=layers)
    corrupt_cache = cache_head_z(loader, corrupt_prompts, layers=layers)

    # Aggregate across last N positions. IOI prompts are typically ~15 tokens;
    # the last 3-4 positions cover the duplicate name and prediction site.
    n_positions = 4

    candidates: list[Candidate] = []
    for layer in layers:
        if layer not in clean_cache or layer not in corrupt_cache:
            continue
        clean_z = clean_cache[layer]   # [B, S, H, d_head]
        corrupt_z = corrupt_cache[layer]
        if clean_z.shape != corrupt_z.shape:
            continue

        # Extract last n_positions: [-4, -3, -2, -1]
        seq_len = clean_z.shape[1]
        start_pos = max(0, seq_len - n_positions)
        clean_z_slice = clean_z[:, start_pos:, :, :]   # [B, N, H, d_head]
        corrupt_z_slice = corrupt_z[:, start_pos:, :, :]

        # Compute |delta| and aggregate across positions and batch
        delta = (clean_z_slice - corrupt_z_slice).abs()  # [B, N, H, d_head]

        # Sum across d_head, mean across positions and batch
        per_head = delta.sum(dim=-1).mean(dim=1).mean(dim=0)  # [H]

        for h in range(per_head.shape[0]):
            candidates.append(
                Candidate(
                    layer=int(layer),
                    idx=int(h),
                    score=float(per_head[h].item()),
                    kind="attn_head",
                    token_pos=-1,  # Represents aggregation over multiple positions
                    metadata={
                        "method": "position_aggregated_z_diff",
                        "n_positions": n_positions,
                    },
                )
            )

    candidates.sort(key=lambda c: -abs(c.score))
    return candidates[:top_k]
