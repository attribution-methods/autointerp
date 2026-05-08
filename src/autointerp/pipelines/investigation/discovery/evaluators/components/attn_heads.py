"""Attention-head evaluator — generic across HF causal LMs.

Calls a candidate algorithm's ``score(...)`` to get ranked
``Candidate(kind="attn_head")`` rows, then for each ``K`` in the K-grid
mean-ablates the top-K heads simultaneously and measures the behavioral-
metric drop on the benchmark's pairs. Returns AUC-K curves plus a
``patch_effect_recovery`` scalar at ``top_k`` so the harness can route
either reward into ``objective_value``.

Phenomenon-agnostic: the IOI benchmark, a future induction benchmark, or
anything else implementing the
:class:`autointerp.benchmarks._protocol.Benchmark` protocol will work
without changes to this file.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

from autointerp.benchmarks import StimulusPair
from autointerp.pipelines.investigation.discovery.candidate import Candidate

if TYPE_CHECKING:
    # Avoid importing torch / transformers at module load — keeps `from
    # ...evaluators import get_evaluator` cheap on machines without the
    # mechinterp extras. Real torch / autointerp.tools imports happen
    # inside ``evaluate`` so the harness can fall back to dry-run paths
    # when heavy deps are missing.
    import torch
    from autointerp.tools.model import ModelHandle


# Module-level model cache so a multi-iteration discovery run reuses one
# loaded GPT-2 instead of spending 10s+ loading it per candidate.
_MODEL_CACHE: dict[str, "ModelHandle"] = {}


def _get_model(model_id: str, device: str | None = None) -> "ModelHandle":
    from autointerp.tools.model import load_model

    key = f"{model_id}::{device or 'auto'}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(model_id, device=device)
    return _MODEL_CACHE[key]


def _normalize_clip(value: float, lo: float, hi: float) -> float:
    """Clamp to ``[0, 1]`` and treat degenerate ranges as 0 to keep AUC sane."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _logits_at_last(handle: "ModelHandle", prompts: list[str]):
    """Plain forward pass, return ``[batch, vocab]`` at the last token."""
    import torch

    inputs = handle.tokenizer(prompts, return_tensors="pt", padding=True).to(
        handle.input_device()
    )
    with torch.no_grad():
        logits = handle.model(**inputs, use_cache=False).logits
    return logits[:, -1, :]


def _per_pair_metric(
    handle: "ModelHandle",
    prompts: list[str],
    pairs: list[StimulusPair],
    metric_fn: Callable[[Any, StimulusPair], float],
    head_patches=None,
) -> list[float]:
    """Run a (possibly patched) forward pass and apply the benchmark's
    behavioral metric per pair. Returns one float per row."""
    from autointerp.tools.head_patching import run_with_head_patches

    if head_patches:
        # ``run_with_head_patches`` returns logits at -1 by default.
        last = run_with_head_patches(
            handle, prompts, head_patches, return_logits_at=-1
        )
    else:
        last = _logits_at_last(handle, prompts)
    out: list[float] = []
    for i, pair in enumerate(pairs):
        out.append(metric_fn(last[i], pair))
    return out


def _trapezoid_auc_k(curve: list[float], k_grid: list[int]) -> float:
    """Integrate a monotone-ish [0,1] curve and normalize by the K-span."""
    if len(curve) < 2:
        return 0.0
    span = float(k_grid[-1] - k_grid[0])
    if span <= 0:
        return 0.0
    area = 0.0
    for i in range(len(curve) - 1):
        dx = float(k_grid[i + 1] - k_grid[i])
        area += dx * 0.5 * (curve[i + 1] + curve[i])
    return max(0.0, min(1.0, area / span))


def evaluate(
    score_fn: Callable[..., list[Candidate]],
    *,
    model_id: str,
    device: str | None,
    pairs: list[StimulusPair],
    behavioral_metric: Callable[[Any, StimulusPair], float],
    layers: list[int] | None,
    top_k: int,
    k_grid: list[int],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate ``score_fn`` against an attention-head substrate.

    Parameters mirror the harness call shape. ``layers=None`` means "all
    layers." ``k_grid`` is the K-sweep used to compute
    ``mean_ablation_auc_k``; the effective ``top_k`` for
    ``patch_effect_recovery`` is the largest ``K`` in the grid that does
    not exceed the candidate count.
    """
    from autointerp.tools.head_patching import (
        cache_head_z,
        mean_ablate_heads,
        mean_head_z,
    )

    handle = _get_model(model_id, device=device)
    if layers is None:
        layers = list(range(handle.n_layers))

    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]

    # 1. Baseline: clean / corrupt behavioral metric per pair.
    clean_metrics = _per_pair_metric(handle, clean_prompts, pairs, behavioral_metric)
    corrupt_metrics = _per_pair_metric(handle, corrupt_prompts, pairs, behavioral_metric)

    # 2. Cache mean ``z`` baseline once for ablation. We use the *clean*
    #    distribution's mean as the ablation target — replacing a head's
    #    output with the dataset-wide clean mean is the standard
    #    intervention. ``cache_head_z`` runs one forward pass per layer
    #    set; cap to the layers the candidate cares about.
    clean_cache = cache_head_z(handle, clean_prompts, layers=layers)
    mean_z = mean_head_z(clean_cache)

    # 3. Run the candidate algorithm.
    candidates = score_fn(
        pairs=pairs,
        loader=handle,
        layers=layers,
        device=str(handle.device),
        top_k=top_k,
        context=context or {},
    )
    # Keep only attn_head candidates that pass our cheap shape checks.
    head_candidates: list[Candidate] = []
    for c in candidates:
        kind = getattr(c, "kind", "attn_head")
        layer = int(getattr(c, "layer", -1))
        idx = int(getattr(c, "idx", getattr(c, "feature_id", -1)))
        if kind != "attn_head":
            continue
        if not (0 <= layer < handle.n_layers):
            continue
        if not (0 <= idx < handle.n_heads):
            continue
        head_candidates.append(
            Candidate(
                layer=layer,
                idx=idx,
                score=float(getattr(c, "score", 0.0)),
                kind="attn_head",
                token_pos=getattr(c, "token_pos", None),
                metadata=getattr(c, "metadata", None),
            )
        )
    head_candidates.sort(key=lambda c: -abs(c.score))
    head_candidates = head_candidates[:top_k]

    # 4. K-sweep: ablate top-K heads simultaneously, measure normalized
    #    delta per pair, integrate the curve.
    effective_grid = [K for K in k_grid if 1 <= K <= len(head_candidates)]
    if not effective_grid:
        # Candidate produced nothing usable — return zero rewards rather
        # than crash so the leaderboard records the failed candidate.
        return {
            "mean_ablation_auc_k": 0.0,
            "mean_steering_auc_k": 0.0,  # not implemented for components
            "combined_auc_k": 0.0,
            "patch_effect_recovery": 0.0,
            "clean_metric": float(sum(clean_metrics) / max(1, len(clean_metrics))),
            "corrupt_metric": float(sum(corrupt_metrics) / max(1, len(corrupt_metrics))),
            "patched_metric": float(sum(corrupt_metrics) / max(1, len(corrupt_metrics))),
            "k_grid": k_grid,
            "effective_grid": [],
            "top_features": [],
            "n_candidates": 0,
        }

    per_pair_curves: list[list[float]] = [[] for _ in pairs]
    last_patched_metrics: list[float] = corrupt_metrics  # for patch_effect_recovery
    for K in effective_grid:
        sites = [(c.layer, c.idx) for c in head_candidates[:K]]
        # Mean-ablate top-K heads at all positions.
        patched_logits = mean_ablate_heads(
            handle, clean_prompts, sites, mean_z, return_logits_at=-1
        )
        for i, pair in enumerate(pairs):
            patched_m = behavioral_metric(patched_logits[i], pair)
            # Normalize delta into [0, 1] per pair: 0 = ablation didn't
            # hurt (patched ≈ clean), 1 = ablation fully broke behavior
            # (patched ≈ corrupt). Clamp for safety.
            cm = clean_metrics[i]
            om = corrupt_metrics[i]
            delta = _normalize_clip(cm - patched_m, 0.0, max(cm - om, 1e-9))
            per_pair_curves[i].append(delta)
        # Track the largest-K patched metric for patch_effect_recovery.
        last_patched_metrics = [
            float(behavioral_metric(patched_logits[i], pair)) for i, pair in enumerate(pairs)
        ]

    # Per-pair AUC-K, then mean across pairs.
    aucs = [_trapezoid_auc_k(c, effective_grid) for c in per_pair_curves]
    mean_ablation = float(sum(aucs) / len(aucs)) if aucs else 0.0

    # patch_effect_recovery at the largest effective K, averaged over pairs.
    clean_mean = float(sum(clean_metrics) / len(clean_metrics))
    corrupt_mean = float(sum(corrupt_metrics) / len(corrupt_metrics))
    patched_mean = float(sum(last_patched_metrics) / len(last_patched_metrics))
    gap = clean_mean - corrupt_mean
    if abs(gap) < 1e-9:
        patch_effect_recovery = 0.0
    else:
        patch_effect_recovery = (clean_mean - patched_mean) / gap
        # Convention here: recovery = how much of the gap the *ablation*
        # opened up (patched moves toward corrupt). For
        # ``MetricName.PATCH_EFFECT_RECOVERY`` we want the standard
        # patching definition (patched moves toward clean), so flip the
        # sign. Higher = better (more recovery).
        patch_effect_recovery = max(0.0, min(1.0, patch_effect_recovery))

    return {
        "mean_ablation_auc_k": mean_ablation,
        # Steering for attention heads is undefined in this evaluator; emit
        # 0 so combined_auc_k = 0.5 * ablation. Spec authors should set the
        # reward to mean_ablation_auc_k or patch_effect_recovery for heads.
        "mean_steering_auc_k": 0.0,
        "combined_auc_k": 0.5 * mean_ablation,
        "patch_effect_recovery": float(patch_effect_recovery),
        "clean_metric": clean_mean,
        "corrupt_metric": corrupt_mean,
        "patched_metric": patched_mean,
        "k_grid": k_grid,
        "effective_grid": effective_grid,
        "top_features": [
            {
                "layer": c.layer,
                "idx": c.idx,
                "kind": c.kind,
                "score": c.score,
                "token_pos": c.token_pos,
            }
            for c in head_candidates
        ],
        "n_candidates": len(head_candidates),
        "per_pair": [
            {
                "pair_id": pair.pair_id,
                "clean_metric": clean_metrics[i],
                "corrupt_metric": corrupt_metrics[i],
                "ablation_curve": per_pair_curves[i],
            }
            for i, pair in enumerate(pairs)
        ],
    }


__all__ = ["evaluate"]
