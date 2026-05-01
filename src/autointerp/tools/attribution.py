"""Gradient and attribution-patching utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .activations import LayerLike, component_module, resolve_layer
from .model import ModelHandle


def next_token_gradient_x_activation(
    handle: ModelHandle,
    prompt: str,
    target_token: str,
    layer: LayerLike,
    component: str = "resid",
    token_index: int = -1,
) -> Dict[str, Any]:
    layer_idx = resolve_layer(layer, handle.n_layers)
    saved: Dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        hidden.retain_grad()
        saved["hidden"] = hidden
        return output

    module = component_module(handle, layer_idx, component)
    hook = module.register_forward_hook(hook_fn)
    try:
        inputs = handle.tokenize(prompt)
        outputs = handle.model(**inputs, use_cache=False)
        target_id = handle.tokenizer.encode(target_token, add_special_tokens=False)[-1]
        logprob = F.log_softmax(outputs.logits[:, -1, :], dim=-1)[0, target_id]
        handle.model.zero_grad(set_to_none=True)
        logprob.backward()
        hidden = saved["hidden"]
        grad = hidden.grad
        scores = (grad * hidden).sum(dim=-1).detach().cpu()[0]
        return {
            "target_token_id": int(target_id),
            "target_token": handle.tokenizer.decode([target_id]),
            "logprob": float(logprob.detach().cpu()),
            "scores_by_position": scores.tolist(),
            "selected_score": float(scores[token_index]),
        }
    finally:
        hook.remove()
        handle.model.zero_grad(set_to_none=True)


def attribution_patch_score(
    clean_activation: torch.Tensor,
    corrupted_activation: torch.Tensor,
    clean_gradient: torch.Tensor,
) -> float:
    delta = clean_activation.detach().cpu() - corrupted_activation.detach().cpu()
    grad = clean_gradient.detach().cpu()
    return float((delta * grad).sum())


def rank_attributions(
    scores: Dict[str, float],
    top_k: Optional[int] = None,
) -> List[Dict[str, float]]:
    rows = [
        {"site": site, "score": float(score), "abs_score": abs(float(score))}
        for site, score in scores.items()
    ]
    rows.sort(key=lambda row: row["abs_score"], reverse=True)
    return rows[:top_k] if top_k is not None else rows
