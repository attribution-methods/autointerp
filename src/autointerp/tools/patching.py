"""Activation patching and causal tracing helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch

from .activations import LayerLike, component_module, resolve_layer
from .model import ModelHandle


def patch_generation(
    handle: ModelHandle,
    prompt: str,
    source_activation: torch.Tensor,
    layer: LayerLike,
    component: str = "resid",
    patch_positions: Sequence[int] = (-1,),
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    layer_idx = resolve_layer(layer, handle.n_layers)
    source = source_activation.to(handle.input_device()).to(handle.dtype)

    def hook_fn(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()
        if hidden.shape[1] == 1:
            return output
        modified = hidden.clone()
        for pos in patch_positions:
            idx = hidden.shape[1] + pos if pos < 0 else pos
            if 0 <= idx < hidden.shape[1]:
                modified[:, idx, :] = source.to(hidden.device).to(hidden.dtype).view(1, -1)
        return (modified,) + rest if isinstance(output, tuple) else modified

    module = component_module(handle, layer_idx, component)
    hook = module.register_forward_hook(hook_fn)
    try:
        return handle.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    finally:
        hook.remove()


def ablate_generation(
    handle: ModelHandle,
    prompt: str,
    layer: LayerLike,
    component: str = "resid",
    positions: Sequence[int] = (-1,),
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    layer_idx = resolve_layer(layer, handle.n_layers)

    def hook_fn(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()
        if hidden.shape[1] == 1:
            return output
        modified = hidden.clone()
        for pos in positions:
            idx = hidden.shape[1] + pos if pos < 0 else pos
            if 0 <= idx < hidden.shape[1]:
                modified[:, idx, :] = 0
        return (modified,) + rest if isinstance(output, tuple) else modified

    module = component_module(handle, layer_idx, component)
    hook = module.register_forward_hook(hook_fn)
    try:
        return handle.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    finally:
        hook.remove()


def sweep_patch_sites(
    handle: ModelHandle,
    prompt: str,
    source_activation: torch.Tensor,
    layers: Iterable[LayerLike],
    components: Iterable[str] = ("resid",),
    patch_positions: Sequence[int] = (-1,),
    max_new_tokens: int = 64,
) -> List[Dict[str, str]]:
    results = []
    for layer in layers:
        layer_idx = resolve_layer(layer, handle.n_layers)
        for component in components:
            text = patch_generation(
                handle,
                prompt,
                source_activation,
                layer_idx,
                component=component,
                patch_positions=patch_positions,
                max_new_tokens=max_new_tokens,
            )
            results.append({"layer": str(layer_idx), "component": component, "output": text})
    return results
