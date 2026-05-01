"""Generation-time interventions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .activations import LayerLike, resolve_layer
from .model import ModelHandle


def generate_with_steering(
    handle: ModelHandle,
    prompt: str,
    layer: LayerLike,
    steering_vector: torch.Tensor,
    strength: float = 1.0,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    steering_start_pos: Optional[int] = None,
    **generation_kwargs: Any,
) -> str:
    layer_idx = resolve_layer(layer, handle.n_layers)
    vector = (steering_vector * strength).to(handle.input_device()).to(handle.dtype)
    tokens_seen = [0]

    def hook_fn(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()
        seq_len = hidden.shape[1]
        start = tokens_seen[0]
        end = start + seq_len
        tokens_seen[0] = end
        vec = vector.to(hidden.device).to(hidden.dtype)
        if steering_start_pos is None or steering_start_pos <= start:
            modified = hidden + vec.view(1, 1, -1)
        elif steering_start_pos >= end:
            return output
        else:
            offset = steering_start_pos - start
            modified = hidden.clone()
            modified[:, offset:, :] += vec.view(1, 1, -1)
        return (modified,) + rest if isinstance(output, tuple) else modified

    hook = handle.layer(layer_idx).register_forward_hook(hook_fn)
    try:
        inputs = handle.tokenize(prompt)
        input_length = inputs["input_ids"].shape[1]
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": handle.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs.update({"do_sample": True, "temperature": temperature})
            kwargs.update(generation_kwargs)
        with torch.no_grad():
            output_ids = handle.model.generate(**inputs, **kwargs)
        new_tokens = output_ids[0][input_length:]
        return handle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    finally:
        hook.remove()


def generate_with_multi_steering(
    handle: ModelHandle,
    prompt: str,
    layer_vectors: Dict[LayerLike, Tuple[torch.Tensor, float]],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    steering_start_pos: Optional[int] = None,
) -> str:
    hooks = []
    counters: Dict[int, List[int]] = {}

    def make_hook(layer_idx: int, vector: torch.Tensor, strength: float):
        vec = (vector * strength).to(handle.input_device()).to(handle.dtype)
        counters[layer_idx] = [0]

        def hook_fn(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            seq_len = hidden.shape[1]
            start = counters[layer_idx][0]
            end = start + seq_len
            counters[layer_idx][0] = end
            if steering_start_pos is not None and steering_start_pos >= end:
                return output
            local_vec = vec.to(hidden.device).to(hidden.dtype)
            modified = hidden + local_vec.view(1, 1, -1)
            return (modified,) + rest if isinstance(output, tuple) else modified

        return hook_fn

    try:
        for layer, (vector, strength) in layer_vectors.items():
            layer_idx = resolve_layer(layer, handle.n_layers)
            hook = handle.layer(layer_idx).register_forward_hook(
                make_hook(layer_idx, vector, strength)
            )
            hooks.append(hook)
        return handle.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    finally:
        for hook in hooks:
            hook.remove()
