"""TEMPLATE for a discovery EVALUATOR (the reward function).

Copy this into your run's ``scripts/`` (or any importable module), fill in the
model / dataset / behavior specifics, and point ``discover_features`` /
``--evaluator`` at it as ``path/to/file.py:evaluate`` or ``module:evaluate``.

The evaluator is the JUDGE: it takes a candidate ``score_fn`` (which ranks
components/features), intervenes on the REAL model, measures the behavioral
effect, and returns a reward in [0, 1]. The candidate proposes; the evaluator
scores. Getting this right is the crux — and the most common failure is a
silent no-op (the intervention doesn't actually change the measured quantity).
This template bakes in the correct boilerplate + a sanity check that catches
that failure.

Contract
--------
    evaluate(score_fn, *, top_k, k_grid, seed=0) -> dict with keys:
      - "mean_ablation_auc_k": float in [0, 1]
      - "mean_steering_auc_k": float in [0, 1]   (set == ablation if single-objective)
      - "top_features": list[{"layer", "idx", "kind", "score"}]
      - "k_grid": list[int]

``score_fn`` signature (what the candidate implements):
    score(pairs, *, loader, layers, device, top_k, context=None) -> list[Candidate]
Pass any precomputed per-component signals to the candidate via ``context``.
"""

from __future__ import annotations

from typing import Any

# ---- Common-bug checklist (the traps we actually hit) ----------------------
# 1. Ablate the RIGHT positions. Editing only the last position cannot change a
#    full-sequence shifted-label loss (HF drops the last-position logit) — and
#    earlier positions can't attend to it. Ablate every position you measure on.
# 2. In a forward hook, ``hook.layer`` is a METHOD, not an int. Use per-layer
#    hook CLOSURES (see _make_ablation_hooks) — don't key a dict on ``hook.layer``.
# 3. Return the modified output. For HF/TL decoder layers the output is a tuple
#    ``(hidden_states, ...)`` — modify ``output[0]`` and return the tuple.
# 4. Measure the metric WHERE the edit lands (e.g. the continuation tokens),
#    not a quantity the edit can't reach.
# 5. SANITY-CHECK that the intervention changed the metric (assert at the end).


def _make_ablation_hooks(by_layer: dict[int, list[int]]):
    """Per-layer forward hooks that mean-ablate the given head indices.

    Returns [(hook_name, hook_fn), ...]. Each hook closes over its own head
    list (do NOT rely on ``hook.layer``).
    """
    def make(head_idxs: list[int]):
        def hook(z, hook):  # z: [batch, pos, n_heads, d_head]
            for h in head_idxs:
                # mean-ablate across the batch, at EVERY position (not just -1)
                z[:, :, h, :] = z[:, :, h, :].mean(dim=0, keepdim=True)
            return z
        return hook
    return [(f"blocks.{layer}.attn.hook_z", make(hs)) for layer, hs in by_layer.items()]


def evaluate(score_fn, *, top_k: int, k_grid: list[int], seed: int = 0) -> dict[str, Any]:
    """Score a candidate ranking by real causal recovery. FILL IN THE TODOs."""
    import torch  # noqa: F401  (heavy imports stay inside the function)


    # --- TODO 1: load the model + build the contrast pairs ------------------
    # from transformer_lens import HookedTransformer
    # model = HookedTransformer.from_pretrained("<model>", device="cuda", dtype="bfloat16")
    # toks, last, target_id, foil_id = ...   # uniform-length batch; track the
    #                                        # position you measure on.
    # m_clean = behavioral_metric(model, toks)   # e.g. logit(target)-logit(foil)
    raise NotImplementedError("Fill in model loading + contrast pairs (TODO 1).")

    # --- TODO 2: precompute per-component signals for the candidate ---------
    # context = {"n_layers": L, "n_heads": H, "head_signals": {
    #     "dla": [...], "attn_to_target": [...], "out_norm": [...]}}
    #
    # --- TODO 3: get the candidate's ranking --------------------------------
    # cands = score_fn([{...} for _ in range(n)], loader=None,
    #                  layers=list(range(L)), device=device,
    #                  top_k=max(top_k, max(k_grid)), context=context)
    # order = dedup_valid([(int(c.layer), int(c.idx)) for c in cands], L, H)
    #
    # --- TODO 4: ablate top-K and measure recovery over the K sweep ---------
    # denom = abs(m_clean) or 1.0
    # curve = []
    # for k in k_grid:
    #     by_layer = group_by_layer(order[:k])
    #     hooks = _make_ablation_hooks(by_layer)
    #     m_abl = behavioral_metric_with_hooks(model, toks, hooks)
    #     curve.append(max(0.0, min(1.0, (m_clean - m_abl) / denom)))
    # auc = _mean_auc_k({"delta_curve_per_pair": [curve],
    #                    "k_grid": [float(k) for k in k_grid]},
    #                   name=MetricName.MEAN_ABLATION_AUC_K)
    #
    # --- TODO 5 (REQUIRED): sanity-check the intervention actually bites ----
    # full = _ablate_metric_all_top_heads(...)   # ablate a big chunk
    # assert abs(full - m_clean) > 1e-4, (
    #     "Intervention had ~zero effect — broken hook / wrong position / wrong "
    #     "metric. Fix the evaluator before trusting the reward.")
    #
    # return {"mean_ablation_auc_k": auc, "mean_steering_auc_k": auc,
    #         "top_features": [...], "k_grid": k_grid}


__all__ = ["evaluate"]
