"""Example evaluator: Qwen2.5-1.5B attention-head ranking for "semantic surprise".

Behavior: after a setup sentence, does the model prefer the SURPRISING
continuation's first token over the UNSURPRISING one. Reward = mean-ablation
recovery of that logit-diff over a top-K head sweep.

Point at it with ``--evaluator
autointerp.pipelines.investigation.discovery.evaluators.surprise:evaluate``.
Needs the ``mechinterp`` extra + a GPU; uses bf16 (~3 GB). Dataset:
``data/semantic_surprise_pairs/dev.jsonl`` (override via AUTOINTERP_SURPRISE_DATA).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEVICE = os.environ.get("AUTOINTERP_DEVICE", "cuda")
_MODEL_ID = os.environ.get("AUTOINTERP_SURPRISE_MODEL", "Qwen/Qwen2.5-1.5B")
_DATA = Path(os.environ.get(
    "AUTOINTERP_SURPRISE_DATA",
    str(Path(__file__).resolve().parents[6] / "data" / "semantic_surprise_pairs" / "dev.jsonl"),
))


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformer_lens import HookedTransformer
    dev = _DEVICE if (_DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    m = HookedTransformer.from_pretrained(_MODEL_ID, device=dev, dtype="bfloat16")
    m.eval()
    return m


def _first_tok(model, text: str) -> int:
    return model.to_tokens(" " + text.strip(), prepend_bos=False)[0, 0].item()


@lru_cache(maxsize=1)
def _setup():
    import torch
    model = _model()
    L, H = model.cfg.n_layers, model.cfg.n_heads
    rows = [json.loads(x) for x in _DATA.read_text().splitlines() if x.strip()]
    items = [(r["setup"], _first_tok(model, r["surprising_continuation"]),
              _first_tok(model, r["unsurprising_continuation"])) for r in rows]
    lens = [model.to_tokens(s).shape[1] for s, _, _ in items]
    modal = max(set(lens), key=lens.count)
    items = [it for it, n in zip(items, lens) if n == modal][:24]

    prompts = [s for s, _, _ in items]
    toks = model.to_tokens(prompts)
    dev = toks.device
    surp = torch.tensor([a for _, a, _ in items], device=dev)
    unsurp = torch.tensor([b for _, _, b in items], device=dev)
    last = toks.shape[1] - 1
    with torch.no_grad():
        logits, cache = model.run_with_cache(toks)
    lg = logits[:, last, :]
    m_clean = (lg.gather(1, surp[:, None]) - lg.gather(1, unsurp[:, None])).mean().item()

    diff_dir = (model.W_U[:, surp] - model.W_U[:, unsurp]).T
    dla = torch.zeros(L, H, device=dev)
    out_norm = torch.zeros(L, H, device=dev)
    attn_max = torch.zeros(L, H, device=dev)
    for layer in range(L):
        z = cache[f"blocks.{layer}.attn.hook_z"][:, last]
        head_out = torch.einsum("bhd,hdm->bhm", z, model.W_O[layer])
        dla[layer] = torch.einsum("bhm,bm->bh", head_out, diff_dir).mean(0)
        out_norm[layer] = head_out.norm(dim=-1).mean(0)
        patt = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, last, :]
        attn_max[layer] = patt.max(dim=-1).values.mean(0)
    return {"model": model, "toks": toks, "last": last, "surp": surp, "unsurp": unsurp,
            "m_clean": m_clean, "dla": dla, "out_norm": out_norm, "attn_max": attn_max,
            "L": L, "H": H, "n": len(prompts)}


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
    return (lg.gather(1, s["surp"][:, None]) - lg.gather(1, s["unsurp"][:, None])).mean().item()


def evaluate(score_fn, *, top_k: int, k_grid: list[int], seed: int = 0) -> dict[str, Any]:
    s = _setup()
    L, H, m_clean = s["L"], s["H"], s["m_clean"]
    context = {"behavior": "semantic_surprise", "n_layers": L, "n_heads": H,
               "head_signals": {"dla": s["dla"].tolist(), "attn_max": s["attn_max"].tolist(),
                                "out_norm": s["out_norm"].tolist()}}
    cands = score_fn([{"i": i} for i in range(s["n"])], loader=None,
                     layers=list(range(L)), device=str(s["toks"].device),
                     top_k=max(top_k, max(k_grid)), context=context)
    seen, order = set(), []
    for c in cands:
        lh = (int(c.layer), int(getattr(c, "idx", 0)))
        if lh not in seen and 0 <= lh[0] < L and 0 <= lh[1] < H:
            seen.add(lh)
            order.append(lh)

    if order:
        probe = _ablate_metric(order[: min(40, len(order))])
        assert abs(probe - m_clean) > 1e-4, (
            f"surprise evaluator: ablation had ~zero effect ({m_clean:.4f}->{probe:.4f}); "
            "silent no-op — fix the hook before trusting the reward.")

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
