"""Activation extraction and component addressing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

from .model import ModelHandle


LayerLike = Union[int, float, str]


@dataclass(frozen=True)
class ComponentSpec:
    layers: List[int]
    component: str = "resid"
    heads: Optional[List[int]] = None


def resolve_layer(layer: LayerLike, n_layers: int) -> int:
    if isinstance(layer, float):
        if not 0 <= layer <= 1:
            raise ValueError("Fractional layers must be in [0, 1]")
        return int(layer * (n_layers - 1))
    if isinstance(layer, str):
        spec = parse_component_spec(layer, n_layers)
        if len(spec.layers) != 1:
            raise ValueError(f"Expected one layer, got {spec.layers}")
        return spec.layers[0]
    if layer < 0:
        return n_layers + layer
    return layer


def parse_component_spec(spec: str, n_layers: Optional[int] = None) -> ComponentSpec:
    text = spec.strip().upper()
    match = re.match(r"^L(\d+(?:\.\d+)?(?:-\d+)?)", text)
    if not match:
        raise ValueError(f"Invalid component spec: {spec}")
    layer_part = match.group(1)
    rest = text[match.end():]

    if "-" in layer_part and "." not in layer_part:
        start, end = layer_part.split("-")
        layers = list(range(int(start), int(end) + 1))
    elif "." in layer_part:
        if n_layers is None:
            raise ValueError("n_layers is required for fractional layer specs")
        value = float(layer_part)
        layers = [int(value * (n_layers - 1)) if value <= 1 else int(value)]
    else:
        layers = [int(layer_part)]

    component = "resid"
    heads = None
    if rest.startswith("MLP"):
        component = "mlp"
        rest = rest[3:]
    elif rest.startswith("ATTN"):
        component = "attn"
        rest = rest[4:]
    elif rest.startswith("RESID"):
        component = "resid"
        rest = rest[5:]
    elif rest.startswith("H"):
        component = "head"

    if rest.startswith("H"):
        component = "head"
        head_match = re.match(r"^H(\d+(?:-\d+)?)", rest)
        if not head_match:
            raise ValueError(f"Invalid head spec in {spec}")
        head_part = head_match.group(1)
        if "-" in head_part:
            start, end = head_part.split("-")
            heads = list(range(int(start), int(end) + 1))
        else:
            heads = [int(head_part)]

    return ComponentSpec(layers=layers, component=component, heads=heads)


def component_module(handle: ModelHandle, layer_idx: int, component: str) -> Any:
    layer = handle.layer(layer_idx)
    if component == "resid":
        return layer
    if component == "mlp":
        for name in ("mlp", "feed_forward", "ffn"):
            if hasattr(layer, name):
                return getattr(layer, name)
    if component in {"attn", "head"}:
        for name in ("self_attn", "attention", "attn"):
            if hasattr(layer, name):
                return getattr(layer, name)
    raise ValueError(f"Could not find {component} module at layer {layer_idx}")


def _select_component_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def capture_activations(
    handle: ModelHandle,
    prompts: List[str],
    layer: LayerLike,
    component: str = "resid",
    token_index: int = -1,
    all_positions: bool = False,
) -> torch.Tensor:
    layer_idx = resolve_layer(layer, handle.n_layers)
    acts: List[torch.Tensor] = []

    def hook_fn(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = _select_component_output(output)
        if all_positions:
            act = hidden.detach().cpu()
        else:
            act = hidden[:, token_index, :].detach().cpu()
        acts.append(act)

    module = component_module(handle, layer_idx, component)
    hook = module.register_forward_hook(hook_fn)
    try:
        inputs = handle.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(handle.input_device())
        with torch.no_grad():
            handle.model(**inputs, use_cache=False)
    finally:
        hook.remove()

    if not acts:
        raise RuntimeError("No activations captured")
    return torch.cat(acts, dim=0)


def cache_components(
    handle: ModelHandle,
    prompts: List[str],
    specs: Iterable[str],
    token_index: int = -1,
) -> Dict[str, torch.Tensor]:
    cache = {}
    for spec_text in specs:
        spec = parse_component_spec(spec_text, handle.n_layers)
        for layer_idx in spec.layers:
            key = f"L{layer_idx}{spec.component}"
            cache[key] = capture_activations(
                handle,
                prompts,
                layer_idx,
                component=spec.component,
                token_index=token_index,
            )
    return cache
