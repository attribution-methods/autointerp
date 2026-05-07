"""Direct causal measurement via head_patch_sweep.

Ranks attention heads by their actual patch_effect_recovery when patching
z-activations from clean to corrupt runs. This directly optimizes the
evaluation metric rather than using a proxy heuristic.

Performs a full L×H sweep once per evaluation, measuring the causal
contribution of each head to the IOI logit_diff metric.
"""

from __future__ import annotations

from typing import Any

NAME = "head_patch_sweep_direct"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list:
    """Rank heads by patch_effect_recovery from head_patch_sweep.

    Performs a full sweep of all (layer, head) positions, patching
    z-activations from clean to corrupt and measuring the recovery of
    logit_diff(IO, S). This directly measures what the evaluation harness
    measures, so the score should perfectly align with the objective.
    """
    import torch
    from autointerp.tools.head_patching import head_patch_sweep
    from autointerp.pipelines.investigation.discovery.candidate import Candidate

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    # Extract target/foil token IDs for logit_diff
    # IOI pairs have .target_token_id and .foil_token_id attributes
    targets = [p.target_token_id for p in pairs]
    foils = [p.foil_token_id for p in pairs]

    # Run the full sweep: patches each (layer, head) from clean->corrupt
    # Returns dict mapping (layer, head) -> patch_effect_recovery
    sweep_results = head_patch_sweep(
        loader=loader,
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        targets=targets,
        foils=foils,
        layers=layers,
    )

    # Convert to Candidate objects
    candidates: list[Candidate] = []
    for (layer, head), recovery in sweep_results.items():
        if layer not in layers:
            continue
        candidates.append(
            Candidate(
                layer=int(layer),
                idx=int(head),
                score=float(recovery),
                kind="attn_head",
                token_pos=-1,
                metadata={"method": "head_patch_sweep_direct"},
            )
        )

    # Sort by absolute score (both positive and negative contributions matter)
    candidates.sort(key=lambda c: -abs(c.score))
    return candidates[:top_k]
