"""Features-substrate baseline algorithm.

Working starting point for feature-substrate discovery: ranks SAE
features by ``mean(|f_clean − f_corrupted|)`` at the prediction
position. Cheap, gradient-free, mirrors the activation-difference
baseline in the circuitbreaker auto_circuit_discovery reference.

Copy this file as ``algorithm_v1.py`` in your session dir and edit
``score(...)`` to try a different attribution method (EAP-IG, token-
specific patching, layer-normalized variants, …).
"""

from __future__ import annotations

from typing import Any

from autointerp.pipelines.investigation.discovery.candidate import Candidate

NAME = "features_baseline_activation_diff"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list[Candidate]:
    """Rank SAE features by ``mean(|f_clean − f_corrupted|)``.

    Requires ``loader.saes`` (dict layer -> SAE) and
    ``loader.hook_context()``. Each forward pass under the hook context
    populates ``.get_activations(layer)`` with shape ``[B, S, n_features]``.
    """
    import torch

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    if not hasattr(loader, "saes") or loader.saes is None:
        raise RuntimeError(
            "features_baseline.score: loader.saes is unset; the spec's "
            "DiscoveryConfig.substrate must be 'features' and the "
            "evaluator must wire SAE hooks before calling score(...)."
        )

    candidates: list[Candidate] = []
    with loader.hook_context() as hooks:
        # Clean pass.
        loader.tokenizer(clean_prompts, return_tensors="pt", padding=True).to(
            loader.input_device()
        )
        with torch.no_grad():
            loader.model(**loader.tokenizer(clean_prompts, return_tensors="pt", padding=True).to(loader.input_device()))
        clean_acts = {layer: hooks.get_activations(layer) for layer in layers}
        # Corrupt pass.
        with torch.no_grad():
            loader.model(**loader.tokenizer(corrupt_prompts, return_tensors="pt", padding=True).to(loader.input_device()))
        corrupt_acts = {layer: hooks.get_activations(layer) for layer in layers}

    for layer in layers:
        if layer not in clean_acts or layer not in corrupt_acts:
            continue
        clean = clean_acts[layer][:, -1, :]    # [B, n_features]
        corrupt = corrupt_acts[layer][:, -1, :]
        if clean.shape != corrupt.shape:
            continue
        per_feat = (clean - corrupt).abs().mean(dim=0)  # [n_features]
        # Top-k per layer to keep memory sane; the registry harness
        # re-merges across layers and trims to global top_k.
        topk = torch.topk(per_feat, k=min(top_k, per_feat.shape[0]))
        for s, i in zip(topk.values.tolist(), topk.indices.tolist()):
            candidates.append(
                Candidate(
                    layer=int(layer),
                    idx=int(i),
                    score=float(s),
                    kind="sae_feature",
                    token_pos=-1,
                    metadata={"method": "activation_diff_at_-1"},
                )
            )

    candidates.sort(key=lambda c: -abs(c.score))
    return candidates[:top_k]
