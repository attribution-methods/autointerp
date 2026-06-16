"""Example evaluator: GPT-2-small attention-head ranking for IOI.

Reward = mean-ablation logit-diff recovery over a top-K sweep. The candidate
ranks heads by combining per-head signals we precompute (dla / attn_to_io /
out_norm); the evaluator mean-ablates the top-K and measures how much of the
IOI logit-diff collapses.

Point at it with ``--evaluator
autointerp.pipelines.investigation.discovery.evaluators.ioi_heads:evaluate``.
Needs the ``mechinterp`` extra (torch + transformer_lens) and a GPU.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

_DEVICE = os.environ.get("AUTOINTERP_DEVICE", "cuda")
_NAMES = [" John", " Mary", " Tom", " Sarah", " Mark", " Anna", " Paul", " Lisa",
          " Mike", " Emma", " David", " Laura", " James", " Karen", " Peter", " Susan"]
_PLACES = [" store", " park", " school", " office"]
_OBJECTS = [" drink", " book", " ball", " ring"]


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformer_lens import HookedTransformer
    dev = _DEVICE if (_DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    m = HookedTransformer.from_pretrained("gpt2", device=dev)
    m.eval()
    return m


@lru_cache(maxsize=1)
def _setup():
    import random

    import torch
    model = _model()
    L, H = model.cfg.n_layers, model.cfg.n_heads
    names = [n for n in _NAMES if model.to_tokens(n, prepend_bos=False).shape[1] == 1]
    rng = random.Random(0)
    prompts, io_names, s_names, seen = [], [], [], set()
    while len(prompts) < 48:
        s, io = rng.sample(names, 2)
        p = (f"When{s} and{io} went to the{rng.choice(_PLACES)},{s} gave the"
             f"{rng.choice(_OBJECTS)} to")
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)
        io_names.append(io)
        s_names.append(s)
    lens = [model.to_tokens(p).shape[1] for p in prompts]
    modal = max(set(lens), key=lens.count)
    keep = [i for i, n in enumerate(lens) if n == modal]
    prompts = [prompts[i] for i in keep]
    io_names = [io_names[i] for i in keep]
    s_names = [s_names[i] for i in keep]

    toks = model.to_tokens(prompts)
    dev = toks.device

    def _t0(n):
        return model.to_tokens(n, prepend_bos=False)[0, 0]
    io_id = torch.tensor([_t0(n) for n in io_names], device=dev)
    s_id = torch.tensor([_t0(n) for n in s_names], device=dev)
    last = toks.shape[1] - 1
    with torch.no_grad():
        logits, cache = model.run_with_cache(toks)
    lg = logits[:, last, :]
    m_clean = (lg.gather(1, io_id[:, None]) - lg.gather(1, s_id[:, None])).mean().item()

    diff_dir = (model.W_U[:, io_id] - model.W_U[:, s_id]).T
    dla = torch.zeros(L, H, device=dev)
    out_norm = torch.zeros(L, H, device=dev)
    attn_to_io = torch.zeros(L, H, device=dev)
    io_pos = (toks == io_id[:, None]).float().argmax(dim=1)
    for layer in range(L):
        z = cache[f"blocks.{layer}.attn.hook_z"][:, last]
        head_out = torch.einsum("bhd,hdm->bhm", z, model.W_O[layer])
        dla[layer] = torch.einsum("bhm,bm->bh", head_out, diff_dir).mean(0)
        out_norm[layer] = head_out.norm(dim=-1).mean(0)
        patt = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, last, :]
        gathered = patt.gather(2, io_pos[:, None, None].expand(-1, H, 1)).squeeze(-1)
        attn_to_io[layer] = gathered.mean(0)
    return {"model": model, "toks": toks, "last": last, "io_id": io_id, "s_id": s_id,
            "m_clean": m_clean, "dla": dla, "out_norm": out_norm, "attn_to_io": attn_to_io,
            "L": L, "H": H}


def _ablate_metric(heads: list[tuple[int, int]]) -> float:
    import torch
    s = _setup()
    model, toks, last = s["model"], s["toks"], s["last"]
    by_layer: dict[int, list[int]] = {}
    for (layer, head) in heads:
        by_layer.setdefault(int(layer), []).append(int(head))

    def make(head_idxs):
        def hook(z, hook):
            for h in head_idxs:
                z[:, :, h, :] = z[:, :, h, :].mean(dim=0, keepdim=True)
            return z
        return hook

    hooks = [(f"blocks.{layer}.attn.hook_z", make(hs)) for layer, hs in by_layer.items()]
    with torch.no_grad():
        logits = model.run_with_hooks(toks, fwd_hooks=hooks)
    lg = logits[:, last, :]
    return (lg.gather(1, s["io_id"][:, None]) - lg.gather(1, s["s_id"][:, None])).mean().item()


def evaluate(score_fn, *, top_k: int, k_grid: list[int], seed: int = 0) -> dict[str, Any]:
    s = _setup()
    L, H, m_clean = s["L"], s["H"], s["m_clean"]
    context = {"behavior": "ioi", "n_layers": L, "n_heads": H,
               "head_signals": {"dla": s["dla"].tolist(), "attn_to_io": s["attn_to_io"].tolist(),
                                "out_norm": s["out_norm"].tolist()}}
    cands = score_fn([{"i": i} for i in range(s["toks"].shape[0])], loader=None,
                     layers=list(range(L)), device=str(s["toks"].device),
                     top_k=max(top_k, max(k_grid)), context=context)
    seen, order = set(), []
    for c in cands:
        lh = (int(c.layer), int(getattr(c, "idx", 0)))
        if lh not in seen and 0 <= lh[0] < L and 0 <= lh[1] < H:
            seen.add(lh)
            order.append(lh)

    # Sanity: ablating a big chunk MUST move the metric, else the hook is a no-op.
    if order:
        probe = _ablate_metric(order[: min(20, len(order))])
        assert abs(probe - m_clean) > 1e-4, (
            f"IOI evaluator: ablation had ~zero effect ({m_clean:.4f}->{probe:.4f}); "
            "the intervention is a silent no-op — fix the hook before trusting the reward.")

    denom = abs(m_clean) if abs(m_clean) > 1e-6 else 1.0
    curve = [max(0.0, min(1.0, (m_clean - _ablate_metric(order[:k])) / denom)) for k in k_grid]
    from autointerp.pipelines.investigation.metrics import _mean_auc_k
    from autointerp.spec import MetricName
    auc = _mean_auc_k({"delta_curve_per_pair": [curve], "k_grid": [float(k) for k in k_grid]},
                      name=MetricName.MEAN_ABLATION_AUC_K)
    return {"mean_ablation_auc_k": auc, "mean_steering_auc_k": auc,
            "top_features": [{"layer": ly, "idx": h, "kind": "attn_head",
                              "score": float(s["dla"][ly, h])} for (ly, h) in order[:top_k]],
            "k_grid": k_grid, "_clean_metric": m_clean, "_curve": curve}


__all__ = ["evaluate"]
