"""Activation patching and causal tracing helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

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


def _token_ids(handle: ModelHandle, tokens: Sequence[str]) -> torch.Tensor:
    ids = [
        handle.tokenizer.encode(token, add_special_tokens=False)[0]
        for token in tokens
    ]
    return torch.tensor(ids, device=handle.input_device())


def _attention_c_proj(handle: ModelHandle, layer_idx: int) -> Any:
    layer = handle.layer(layer_idx)
    attn = getattr(layer, "attn", None)
    if attn is None:
        attn = getattr(layer, "attention", None)
    if attn is None:
        attn = getattr(layer, "self_attn", None)
    if attn is None or not hasattr(attn, "c_proj"):
        raise ValueError(
            "head-level patching currently requires a GPT-style attention "
            f"module with c_proj at layer {layer_idx}"
        )
    return attn.c_proj


def _batch_logit_diff(
    handle: ModelHandle,
    logits: torch.Tensor,
    io_tokens: Sequence[str],
    s_tokens: Sequence[str],
    token_index: int,
) -> torch.Tensor:
    io_ids = _token_ids(handle, io_tokens)
    s_ids = _token_ids(handle, s_tokens)
    rows = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[:, token_index, :]
    return final_logits[rows, io_ids] - final_logits[rows, s_ids]


def _paired_inputs(handle: ModelHandle, clean: Sequence[str], corrupt: Sequence[str]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n = len(clean)
    encoded = handle.tokenizer(
        list(clean) + list(corrupt),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(handle.input_device())
    clean_inputs = {key: value[:n] for key, value in encoded.items()}
    corrupt_inputs = {key: value[n:] for key, value in encoded.items()}
    return clean_inputs, corrupt_inputs


def _mean_clean_corrupt_metrics(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    io_tokens: Sequence[str],
    s_tokens: Sequence[str],
    *,
    batch_size: int,
    token_index: int,
) -> tuple[float, float]:
    clean_total = 0.0
    corrupt_total = 0.0
    n_total = 0
    for start in range(0, len(clean_prompts), batch_size):
        end = start + batch_size
        clean = clean_prompts[start:end]
        corrupt = corrupt_prompts[start:end]
        ios = io_tokens[start:end]
        subjects = s_tokens[start:end]
        clean_inputs, corrupt_inputs = _paired_inputs(handle, clean, corrupt)
        with torch.inference_mode():
            clean_logits = handle.model(**clean_inputs, use_cache=False).logits
            corrupt_logits = handle.model(**corrupt_inputs, use_cache=False).logits
        clean_diffs = _batch_logit_diff(handle, clean_logits, ios, subjects, token_index)
        corrupt_diffs = _batch_logit_diff(handle, corrupt_logits, ios, subjects, token_index)
        clean_total += float(clean_diffs.float().sum().item())
        corrupt_total += float(corrupt_diffs.float().sum().item())
        n_total += len(clean)
    return clean_total / n_total, corrupt_total / n_total


def _capture_clean_head_source(
    handle: ModelHandle,
    layer_idx: int,
    clean_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def capture_hook(_module: Any, inputs: Any) -> None:
        captured.append(inputs[0].detach())

    hook = _attention_c_proj(handle, layer_idx).register_forward_pre_hook(capture_hook)
    try:
        with torch.inference_mode():
            handle.model(**clean_inputs, use_cache=False)
    finally:
        hook.remove()
    if not captured:
        raise RuntimeError(f"did not capture attention head source at layer {layer_idx}")
    return captured[0]


def sweep_head_patch_recovery(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    io_tokens: Sequence[str],
    s_tokens: Sequence[str],
    *,
    layers: Iterable[int] | None = None,
    heads: Iterable[int] | None = None,
    batch_size: int = 8,
    token_index: int = -1,
) -> dict[str, Any]:
    """Sweep clean-to-corrupt patches for every GPT-style attention head.

    The patched activation is each head's input to the attention output
    projection (`attn.c_proj`) at `token_index`. For each `(layer, head)`, the
    returned patched metric is mean final-position IO-vs-S logit diff after
    replacing the corrupt run's head activation with the paired clean run's
    activation.
    """
    if not (
        len(clean_prompts)
        == len(corrupt_prompts)
        == len(io_tokens)
        == len(s_tokens)
    ):
        raise ValueError("clean, corrupt, io_tokens, and s_tokens must have equal length")
    if not clean_prompts:
        raise ValueError("at least one prompt pair is required")

    layer_indices = list(layers if layers is not None else range(handle.n_layers))
    head_indices = list(heads if heads is not None else range(handle.n_heads))
    d_head = handle.d_model // handle.n_heads
    clean_metric, corrupt_metric = _mean_clean_corrupt_metrics(
        handle,
        clean_prompts,
        corrupt_prompts,
        io_tokens,
        s_tokens,
        batch_size=batch_size,
        token_index=token_index,
    )
    denom = clean_metric - corrupt_metric

    patched_sum = torch.zeros(
        len(layer_indices),
        len(head_indices),
        dtype=torch.float64,
    )
    n_total = 0

    for start in range(0, len(clean_prompts), batch_size):
        end = start + batch_size
        clean = clean_prompts[start:end]
        corrupt = corrupt_prompts[start:end]
        ios = io_tokens[start:end]
        subjects = s_tokens[start:end]
        clean_inputs, corrupt_inputs = _paired_inputs(handle, clean, corrupt)
        batch_n = len(clean)
        n_total += batch_n

        repeated_inputs = {
            key: value.repeat_interleave(len(head_indices), dim=0)
            for key, value in corrupt_inputs.items()
        }
        repeated_ios = [
            token
            for token in ios
            for _ in head_indices
        ]
        repeated_subjects = [
            token
            for token in subjects
            for _ in head_indices
        ]
        repeated_head_ids = torch.tensor(
            head_indices * batch_n,
            device=handle.input_device(),
        )

        for layer_pos, layer_idx in enumerate(layer_indices):
            source = _capture_clean_head_source(handle, layer_idx, clean_inputs)
            source_repeated = source.repeat_interleave(len(head_indices), dim=0)

            def patch_hook(_module: Any, inputs: Any) -> tuple[torch.Tensor, ...]:
                hidden = inputs[0]
                patched = hidden.clone()
                pos = hidden.shape[1] + token_index if token_index < 0 else token_index
                for head_pos, head_idx in enumerate(head_indices):
                    del head_pos
                    rows = repeated_head_ids == head_idx
                    head_start = head_idx * d_head
                    head_end = head_start + d_head
                    patched[rows, pos, head_start:head_end] = source_repeated[
                        rows,
                        pos,
                        head_start:head_end,
                    ].to(hidden.dtype)
                return (patched,) + tuple(inputs[1:])

            hook = _attention_c_proj(handle, layer_idx).register_forward_pre_hook(patch_hook)
            try:
                with torch.inference_mode():
                    logits = handle.model(**repeated_inputs, use_cache=False).logits
            finally:
                hook.remove()
            diffs = _batch_logit_diff(
                handle,
                logits,
                repeated_ios,
                repeated_subjects,
                token_index,
            ).float()
            patched_sum[layer_pos] += diffs.detach().cpu().reshape(batch_n, len(head_indices)).sum(dim=0)

    patched_metric = patched_sum / n_total
    if abs(denom) < 1e-9:
        recovery = torch.full_like(patched_metric, float("nan"))
    else:
        recovery = (patched_metric - corrupt_metric) / denom

    rows: list[dict[str, float | int]] = []
    for layer_pos, layer_idx in enumerate(layer_indices):
        for head_pos, head_idx in enumerate(head_indices):
            rows.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "patched_metric": float(patched_metric[layer_pos, head_pos].item()),
                    "recovery": float(recovery[layer_pos, head_pos].item()),
                }
            )
    rows.sort(key=lambda row: float(row["recovery"]), reverse=True)
    return {
        "clean_metric": float(clean_metric),
        "corrupt_metric": float(corrupt_metric),
        "patched_metric_by_head": patched_metric.tolist(),
        "recovery_by_head": recovery.tolist(),
        "layers": [int(x) for x in layer_indices],
        "heads": [int(x) for x in head_indices],
        "top_heads": rows,
        "n_samples": n_total,
        "token_index": token_index,
    }


def patch_head_group_metric(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    io_tokens: Sequence[str],
    s_tokens: Sequence[str],
    sites: Sequence[dict[str, int]],
    *,
    batch_size: int = 8,
    token_index: int = -1,
) -> dict[str, Any]:
    """Patch a group of heads and return mean logit diff plus KL to clean."""
    by_layer: dict[int, list[int]] = {}
    for site in sites:
        by_layer.setdefault(int(site["layer"]), []).append(int(site["head"]))
    d_head = handle.d_model // handle.n_heads
    patched_total = 0.0
    n_total = 0
    kl_per_sample: list[float] = []

    for start in range(0, len(clean_prompts), batch_size):
        end = start + batch_size
        clean = clean_prompts[start:end]
        corrupt = corrupt_prompts[start:end]
        ios = io_tokens[start:end]
        subjects = s_tokens[start:end]
        clean_inputs, corrupt_inputs = _paired_inputs(handle, clean, corrupt)
        sources = {
            layer_idx: _capture_clean_head_source(handle, layer_idx, clean_inputs)
            for layer_idx in by_layer
        }
        hooks = []
        try:
            for layer_idx, patched_heads in by_layer.items():
                source = sources[layer_idx]

                def make_hook(layer: int, heads_for_layer: list[int], source_for_layer: torch.Tensor):
                    del layer

                    def patch_hook(_module: Any, inputs: Any) -> tuple[torch.Tensor, ...]:
                        hidden = inputs[0]
                        patched = hidden.clone()
                        pos = hidden.shape[1] + token_index if token_index < 0 else token_index
                        for head_idx in heads_for_layer:
                            head_start = head_idx * d_head
                            head_end = head_start + d_head
                            patched[:, pos, head_start:head_end] = source_for_layer[
                                :,
                                pos,
                                head_start:head_end,
                            ].to(hidden.dtype)
                        return (patched,) + tuple(inputs[1:])

                    return patch_hook

                hooks.append(
                    _attention_c_proj(handle, layer_idx).register_forward_pre_hook(
                        make_hook(layer_idx, patched_heads, source)
                    )
                )
            with torch.inference_mode():
                clean_logits = handle.model(**clean_inputs, use_cache=False).logits
                patched_logits = handle.model(**corrupt_inputs, use_cache=False).logits
        finally:
            for hook in hooks:
                hook.remove()

        diffs = _batch_logit_diff(handle, patched_logits, ios, subjects, token_index).float()
        patched_total += float(diffs.sum().item())
        n_total += len(clean)

        clean_log_probs = F.log_softmax(clean_logits[:, token_index, :].float(), dim=-1)
        patched_probs = F.softmax(patched_logits[:, token_index, :].float(), dim=-1)
        kl = F.kl_div(clean_log_probs, patched_probs, reduction="none").sum(dim=-1)
        kl_per_sample.extend(float(x) for x in kl.detach().cpu().tolist())

    return {
        "patched_metric": patched_total / n_total,
        "kl_per_sample": kl_per_sample,
        "n_samples": n_total,
        "sites": [
            {"layer": int(site["layer"]), "head": int(site["head"])}
            for site in sites
        ],
    }
