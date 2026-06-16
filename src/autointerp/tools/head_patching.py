"""Granular attention-head patching, ablation, and path patching.

The helpers in :mod:`patching` operate at residual / whole-layer / whole-attention
granularity. This module adds the head-level primitives:

- :func:`cache_head_z` captures per-head pre-output-projection activations
  ``z`` (the head outputs before ``W_O``) at chosen layers.
- :func:`run_with_head_patches` runs a forward pass while overriding ``z`` at
  arbitrary ``(layer, head, position)`` sites.
- :func:`mean_ablate_heads` ablates heads by replacing their ``z`` with a
  precomputed dataset mean (the standard ablation baseline).
- :func:`head_patch_sweep` runs a ``(layer, head)`` sweep that patches the
  clean head ``z`` into a corrupt forward pass and returns a ``[L, H]``
  effect matrix.
- :func:`path_patch` performs single-step path patching: a sender head's
  ``z`` is set to its clean value while every other head is frozen to its
  corrupt value, isolating the sender's direct effect on the model logits.
- :func:`logit_diff` is a small batched metric over paired target/contrast
  token ids, used by patching sweeps.

These helpers are model-family-agnostic: they hook the attention output
projection (``c_proj`` for GPT-2 family, ``o_proj`` for Llama/Qwen/Mistral/
Gemma), whose input is the per-position concatenation of head outputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .activations import component_module, resolve_layer
from .model import ModelHandle

HeadSite = Tuple[int, int]  # (layer, head)
HeadPosSite = Tuple[int, int, int]  # (layer, head, position)


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------


def attention_output_projection(handle: ModelHandle, layer_idx: int) -> torch.nn.Module:
    """Return the attention output projection ``W_O`` module for ``layer_idx``.

    Its *input* is the per-position concatenation of head outputs, shape
    ``[batch, seq, n_heads * d_head]``. We hook this rather than the full
    attention block because it is the cleanest, family-independent place to
    intervene at head granularity.
    """
    attn = component_module(handle, layer_idx, "attn")
    for name in ("o_proj", "c_proj", "out_proj", "dense"):
        proj = getattr(attn, name, None)
        if proj is not None:
            return proj
    raise ValueError(f"No attention output projection found at layer {layer_idx}")


def _head_dim(handle: ModelHandle) -> int:
    config = handle.model.config
    for name in ("head_dim", "head_size"):
        if hasattr(config, name):
            return int(getattr(config, name))
    return handle.d_model // handle.n_heads


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------


@dataclass
class _HookHandle:
    handle: Any

    def remove(self) -> None:
        self.handle.remove()


@contextmanager
def _install_pre_hook(module: torch.nn.Module, fn: Callable[[Any, Tuple], Optional[Tuple]]):
    handle = module.register_forward_pre_hook(fn)
    try:
        yield
    finally:
        handle.remove()


def _reshape_to_heads(z_flat: torch.Tensor, n_heads: int, d_head: int) -> torch.Tensor:
    return z_flat.view(*z_flat.shape[:-1], n_heads, d_head)


def _reshape_from_heads(z_heads: torch.Tensor) -> torch.Tensor:
    return z_heads.reshape(*z_heads.shape[:-2], -1)


# ---------------------------------------------------------------------------
# Caching head z
# ---------------------------------------------------------------------------


def _tokenize_batch(handle: ModelHandle, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
    return handle.tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    ).to(handle.input_device())


def cache_head_z(
    handle: ModelHandle,
    prompts: Sequence[str],
    layers: Optional[Iterable[int]] = None,
) -> Dict[int, torch.Tensor]:
    """Run a forward pass and cache per-head ``z`` at each requested layer.

    Returns a dict mapping ``layer_idx -> tensor of shape [batch, seq, n_heads,
    d_head]`` on CPU. ``z`` is the input to the attention output projection
    (``W_O``); equivalently the per-head output before mixing.
    """
    if layers is None:
        layer_indices = list(range(handle.n_layers))
    else:
        layer_indices = [resolve_layer(int(layer), handle.n_layers) for layer in layers]
    n_heads = handle.n_heads
    d_head = _head_dim(handle)

    cache: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def pre_hook(_module: Any, args: Tuple) -> None:
            z_flat = args[0]
            cache[layer_idx] = _reshape_to_heads(z_flat, n_heads, d_head).detach().cpu()
        return pre_hook

    try:
        for layer_idx in layer_indices:
            proj = attention_output_projection(handle, layer_idx)
            hooks.append(proj.register_forward_pre_hook(make_hook(layer_idx)))
        inputs = _tokenize_batch(handle, prompts)
        with torch.no_grad():
            handle.model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return cache


def mean_head_z(cache: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """Average per-head ``z`` over the batch dimension, producing the standard
    mean-ablation baseline ``[seq, n_heads, d_head]`` per layer."""
    return {layer: z.mean(dim=0) for layer, z in cache.items()}


# ---------------------------------------------------------------------------
# Patched forward pass
# ---------------------------------------------------------------------------


@dataclass
class HeadPatch:
    """Override for one ``(layer, head)`` site.

    ``source`` is broadcastable to ``[batch, seq, d_head]``. ``positions``
    selects which token positions to patch (negative indices supported); pass
    ``None`` to patch every position.
    """

    layer: int
    head: int
    source: torch.Tensor
    positions: Optional[Sequence[int]] = None


def _patches_by_layer(patches: Sequence[HeadPatch]) -> Dict[int, List[HeadPatch]]:
    grouped: Dict[int, List[HeadPatch]] = {}
    for p in patches:
        grouped.setdefault(p.layer, []).append(p)
    return grouped


def run_with_head_patches(
    handle: ModelHandle,
    prompts: Sequence[str],
    patches: Sequence[HeadPatch],
    return_logits_at: Optional[int] = -1,
) -> torch.Tensor:
    """Run a forward pass with the given head patches applied.

    Returns the logits tensor. With ``return_logits_at`` set (default ``-1``)
    only the logits at that token position are returned, shape ``[batch,
    vocab]``. Pass ``None`` to get the full ``[batch, seq, vocab]`` tensor.
    """
    n_heads = handle.n_heads
    d_head = _head_dim(handle)
    grouped = _patches_by_layer(patches)
    hooks = []

    def make_hook(layer_patches: List[HeadPatch]):
        def pre_hook(_module: Any, args: Tuple) -> Tuple:
            z_flat = args[0]
            z = _reshape_to_heads(z_flat, n_heads, d_head).clone()
            seq_len = z.shape[1]
            for patch in layer_patches:
                src = patch.source.to(z.device).to(z.dtype)
                if src.dim() == 1:
                    src = src.view(1, 1, -1).expand(z.shape[0], seq_len, -1)
                elif src.dim() == 2:
                    src = src.unsqueeze(0).expand(z.shape[0], -1, -1)
                if patch.positions is None:
                    z[:, :, patch.head, :] = src[:, :seq_len, :]
                else:
                    for pos in patch.positions:
                        idx = seq_len + pos if pos < 0 else pos
                        if 0 <= idx < seq_len:
                            z[:, idx, patch.head, :] = src[:, idx, :]
            return (_reshape_from_heads(z),) + args[1:]
        return pre_hook

    try:
        for layer_idx, layer_patches in grouped.items():
            proj = attention_output_projection(handle, layer_idx)
            hooks.append(proj.register_forward_pre_hook(make_hook(layer_patches)))
        inputs = _tokenize_batch(handle, prompts)
        with torch.no_grad():
            outputs = handle.model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    logits = outputs.logits
    if return_logits_at is None:
        return logits
    return logits[:, return_logits_at, :]


# ---------------------------------------------------------------------------
# Mean ablation
# ---------------------------------------------------------------------------


def mean_ablate_heads(
    handle: ModelHandle,
    prompts: Sequence[str],
    sites: Sequence[HeadSite],
    mean_z: Dict[int, torch.Tensor],
    positions: Optional[Sequence[int]] = None,
    return_logits_at: Optional[int] = -1,
) -> torch.Tensor:
    """Replace each ``(layer, head)`` site's ``z`` with its dataset mean.

    ``mean_z`` is the ``mean_head_z`` of a baseline cache (typically the
    same task distribution).
    """
    patches = []
    for layer, head in sites:
        if layer not in mean_z:
            raise KeyError(f"mean_z is missing layer {layer}")
        patches.append(
            HeadPatch(
                layer=layer,
                head=head,
                source=mean_z[layer][:, head, :],
                positions=positions,
            )
        )
    return run_with_head_patches(handle, prompts, patches, return_logits_at=return_logits_at)


# ---------------------------------------------------------------------------
# Metrics + sweeps
# ---------------------------------------------------------------------------


def logit_diff(
    logits: torch.Tensor,
    target_token_ids: Sequence[int],
    contrast_token_ids: Sequence[int],
) -> torch.Tensor:
    """Compute ``logit(target) - logit(contrast)`` per row.

    ``logits`` has shape ``[batch, vocab]``. ``target_token_ids`` and
    ``contrast_token_ids`` are length-``batch`` sequences of vocab indices.
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected [batch, vocab] logits, got {logits.shape}")
    tgt = torch.tensor(list(target_token_ids), device=logits.device, dtype=torch.long)
    ctr = torch.tensor(list(contrast_token_ids), device=logits.device, dtype=torch.long)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, tgt] - logits[rows, ctr]


def head_patch_sweep(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    metric: Callable[[torch.Tensor], torch.Tensor],
    layers: Optional[Iterable[int]] = None,
    heads: Optional[Iterable[int]] = None,
    patch_positions: Optional[Sequence[int]] = None,
    clean_cache: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Patch each ``(layer, head)`` from clean into a corrupt forward pass.

    For each site, runs the corrupt prompts with that head's ``z`` replaced by
    the clean cache, evaluates ``metric(logits_at_-1)`` (a per-row tensor),
    and records the mean. Also records the unpatched clean and corrupt
    metrics. Returns a dict with the effect matrix ``[L, H]`` indexed by
    chosen layers/heads.
    """
    layer_indices = (
        list(range(handle.n_layers))
        if layers is None
        else [resolve_layer(int(layer), handle.n_layers) for layer in layers]
    )
    head_indices = list(heads) if heads is not None else list(range(handle.n_heads))

    if clean_cache is None:
        clean_cache = cache_head_z(handle, clean_prompts, layers=layer_indices)

    inputs_clean = _tokenize_batch(handle, clean_prompts)
    inputs_corrupt = _tokenize_batch(handle, corrupt_prompts)
    with torch.no_grad():
        clean_logits = handle.model(**inputs_clean, use_cache=False).logits[:, -1, :]
        corrupt_logits = handle.model(**inputs_corrupt, use_cache=False).logits[:, -1, :]
    clean_metric = float(metric(clean_logits).mean())
    corrupt_metric = float(metric(corrupt_logits).mean())

    effects = torch.zeros(len(layer_indices), len(head_indices))
    recovery = torch.zeros_like(effects)
    gap = clean_metric - corrupt_metric

    for li, layer in enumerate(layer_indices):
        z_layer = clean_cache[layer]  # [B, S, H, d_head]
        for hi, head in enumerate(head_indices):
            patch = HeadPatch(
                layer=layer,
                head=head,
                source=z_layer[:, :, head, :],
                positions=patch_positions,
            )
            patched_logits = run_with_head_patches(
                handle, corrupt_prompts, [patch], return_logits_at=-1
            )
            m = float(metric(patched_logits).mean())
            effects[li, hi] = m
            recovery[li, hi] = (m - corrupt_metric) / gap if gap != 0 else 0.0

    return {
        "layers": layer_indices,
        "heads": head_indices,
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
        "patched_metric": effects,
        "recovery": recovery,
    }


# ---------------------------------------------------------------------------
# Path patching (single-step, direct effect on logits)
# ---------------------------------------------------------------------------


def path_patch(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    sender: HeadSite,
    metric: Callable[[torch.Tensor], torch.Tensor],
    patch_positions: Optional[Sequence[int]] = None,
    freeze_layers: Optional[Iterable[int]] = None,
    clean_cache: Optional[Dict[int, torch.Tensor]] = None,
    corrupt_cache: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Single-step path patching: sender's direct effect on the model logits.

    Protocol (Wang et al. 2022, simplified):

    1. Cache clean ``z`` at the sender layer and corrupt ``z`` at every
       layer that should be frozen on the path (``freeze_layers``, defaults
       to all layers other than the sender's).
    2. Run the corrupt prompts with: sender head ``z`` overridden to the
       clean value, every other head's ``z`` overridden to its corrupt
       value (so receiver inputs match the unpatched corrupt run except
       through the sender).
    3. Measure ``metric`` on the resulting logits.

    The reported ``recovery`` is ``(metric_patched - metric_corrupt) /
    (metric_clean - metric_corrupt)``.
    """
    sender_layer, sender_head = sender
    if freeze_layers is None:
        freeze_layers = [
            layer for layer in range(handle.n_layers) if layer != sender_layer
        ]
    freeze_layers = [
        resolve_layer(int(layer), handle.n_layers) for layer in freeze_layers
    ]

    needed = sorted(set([sender_layer, *freeze_layers]))
    if clean_cache is None:
        clean_cache = cache_head_z(handle, clean_prompts, layers=[sender_layer])
    if corrupt_cache is None:
        corrupt_cache = cache_head_z(handle, corrupt_prompts, layers=needed)

    patches: List[HeadPatch] = [
        HeadPatch(
            layer=sender_layer,
            head=sender_head,
            source=clean_cache[sender_layer][:, :, sender_head, :],
            positions=patch_positions,
        )
    ]
    for layer in freeze_layers:
        for head in range(handle.n_heads):
            if layer == sender_layer and head == sender_head:
                continue
            patches.append(
                HeadPatch(
                    layer=layer,
                    head=head,
                    source=corrupt_cache[layer][:, :, head, :],
                    positions=None,
                )
            )

    inputs_clean = _tokenize_batch(handle, clean_prompts)
    inputs_corrupt = _tokenize_batch(handle, corrupt_prompts)
    with torch.no_grad():
        clean_logits = handle.model(**inputs_clean, use_cache=False).logits[:, -1, :]
        corrupt_logits = handle.model(**inputs_corrupt, use_cache=False).logits[:, -1, :]
    patched_logits = run_with_head_patches(handle, corrupt_prompts, patches, return_logits_at=-1)

    clean_m = float(metric(clean_logits).mean())
    corrupt_m = float(metric(corrupt_logits).mean())
    patched_m = float(metric(patched_logits).mean())
    gap = clean_m - corrupt_m
    return {
        "sender": sender,
        "clean_metric": clean_m,
        "corrupt_metric": corrupt_m,
        "patched_metric": patched_m,
        "recovery": (patched_m - corrupt_m) / gap if gap != 0 else 0.0,
    }
