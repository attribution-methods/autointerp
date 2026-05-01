"""Logit lens and direct logit attribution helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .activations import LayerLike, resolve_layer
from .model import ModelHandle


def _maybe_final_norm(handle: ModelHandle, hidden: torch.Tensor) -> torch.Tensor:
    inner = getattr(handle.model, "model", None)
    candidates = [
        getattr(inner, "norm", None),
        getattr(inner, "final_layernorm", None),
        getattr(inner, "ln_f", None),
    ]
    for norm in candidates:
        if norm is not None:
            return norm(hidden)
    return hidden


def unembed(
    handle: ModelHandle,
    hidden: torch.Tensor,
    apply_final_norm: bool = True,
) -> torch.Tensor:
    if apply_final_norm:
        hidden = _maybe_final_norm(handle, hidden)
    if hasattr(handle.model, "lm_head"):
        return handle.model.lm_head(hidden)
    embeddings = handle.model.get_output_embeddings()
    return embeddings(hidden)


def top_tokens(
    handle: ModelHandle,
    logits: torch.Tensor,
    top_k: int = 10,
) -> List[Dict[str, float]]:
    values, ids = torch.topk(logits.detach().cpu(), k=top_k)
    return [
        {
            "token": handle.tokenizer.decode([int(token_id)]),
            "token_id": int(token_id),
            "logit": float(value),
        }
        for value, token_id in zip(values, ids)
    ]


def logit_lens(
    handle: ModelHandle,
    prompt: str,
    layers: Optional[Iterable[LayerLike]] = None,
    token_index: int = -1,
    top_k: int = 10,
    apply_final_norm: bool = True,
    include_kl_to_final: bool = True,
) -> List[Dict[str, Any]]:
    layer_indices = [
        resolve_layer(layer, handle.n_layers) for layer in (layers or range(handle.n_layers))
    ]
    wanted = set(layer_indices)
    captured: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[:, token_index, :].detach()
        return hook_fn

    try:
        for layer_idx in layer_indices:
            hooks.append(handle.layer(layer_idx).register_forward_hook(make_hook(layer_idx)))
        inputs = handle.tokenize(prompt)
        with torch.no_grad():
            outputs = handle.model(**inputs, use_cache=False)
        final_logits = outputs.logits[:, token_index, :]
    finally:
        for hook in hooks:
            hook.remove()

    final_probs = F.softmax(final_logits[0], dim=-1)
    rows = []
    for layer_idx in layer_indices:
        if layer_idx not in wanted or layer_idx not in captured:
            continue
        logits = unembed(handle, captured[layer_idx], apply_final_norm=apply_final_norm)[0]
        row: Dict[str, Any] = {"layer": layer_idx, "top_tokens": top_tokens(handle, logits, top_k)}
        if include_kl_to_final:
            layer_log_probs = F.log_softmax(logits, dim=-1)
            row["kl_to_final"] = float(F.kl_div(layer_log_probs, final_probs, reduction="sum"))
        rows.append(row)
    return rows


def direct_logit_attribution(
    handle: ModelHandle,
    component_vectors: Dict[str, torch.Tensor],
    target_token: str,
    contrast_token: Optional[str] = None,
) -> Dict[str, float]:
    target_id = handle.tokenizer.encode(target_token, add_special_tokens=False)[-1]
    unembed_matrix = handle.model.get_output_embeddings().weight.detach().cpu()
    direction = unembed_matrix[target_id]
    if contrast_token is not None:
        contrast_id = handle.tokenizer.encode(contrast_token, add_special_tokens=False)[-1]
        direction = direction - unembed_matrix[contrast_id]
    return {
        name: float(torch.dot(vector.detach().cpu().flatten(), direction.flatten()))
        for name, vector in component_vectors.items()
    }
