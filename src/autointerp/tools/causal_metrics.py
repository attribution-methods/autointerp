"""One-call bridges from the patching tools to a causal-metric capture.

Running a clean/corrupt/patched pass (``head_patching.path_patch`` /
``head_patch_sweep``) and recording a ``model_forward`` capture
(``provenance.record_capture``) both already exist — but agents kept stalling at
the seam between them, unable to turn a patch sweep into the gate-passing capture
that ``compute_and_commit_metric(metric="patch_effect_recovery", ...)`` needs.

These helpers close that seam: each runs the real forward passes and records the
scalar inputs as a ``model_forward`` capture in one call, returning the relpath
to hand straight to ``compute_and_commit_metric``. They add no new patching math —
they compose existing primitives — so a causal criterion never has to be
abandoned for "I couldn't produce captures". General to any clean/corrupt/patched
study (IOI, induction, CoT, …), not tied to any one question.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch

from autointerp.tools.head_patching import HeadSite, head_patch_sweep, path_patch
from autointerp.tools.model import ModelHandle
from autointerp.tools.provenance import record_capture


def best_patch_site(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    layers: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
) -> HeadSite:
    """Sweep heads and return the ``(layer, head)`` with the highest recovery — a
    candidate ``sender`` for :func:`patch_recovery_capture`.

    ``metric`` maps ``[batch, vocab]`` final-token logits to ``[batch]`` (e.g. a
    target-minus-foil logit difference).
    """
    sweep = head_patch_sweep(
        handle, clean_prompts, corrupt_prompts, metric, layers=layers, heads=heads
    )
    recovery = sweep["recovery"]  # [n_layers, n_heads] tensor over chosen indices
    flat = int(torch.argmax(recovery))
    n_heads = recovery.shape[1]
    li, hi = flat // n_heads, flat % n_heads
    return (sweep["layers"][li], sweep["heads"][hi])


def patch_recovery_capture(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    sender: HeadSite,
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    name: str = "patch_effect_recovery_inputs",
    patch_positions: Optional[Sequence[int]] = (-1,),
    model_id: Optional[str] = None,
    prompt_batch: Optional[str] = None,
) -> str:
    """Run clean/corrupt/patched passes for ``sender`` and record the scalar
    triple ``patch_effect_recovery`` needs as a ``model_forward`` capture.

    Returns the capture relpath — pass it straight to
    ``compute_and_commit_metric(metric="patch_effect_recovery", inputs=<relpath>,
    criterion_id=..., ...)``. ``metric`` maps ``[batch, vocab]`` final-token
    logits to ``[batch]``. ``sender`` is a ``(layer, head)`` site (e.g. from
    :func:`best_patch_site`).
    """
    out = path_patch(
        handle,
        clean_prompts,
        corrupt_prompts,
        sender,
        metric,
        patch_positions=list(patch_positions) if patch_positions else None,
    )
    return record_capture(
        name,
        {
            "clean_metric": float(out["clean_metric"]),
            "corrupt_metric": float(out["corrupt_metric"]),
            "patched_metric": float(out["patched_metric"]),
            "recovery": float(out["recovery"]),
            "sender": [int(sender[0]), int(sender[1])],
        },
        source="model_forward",
        model_id=model_id or getattr(handle, "model_id", None),
        prompt_batch=prompt_batch,
    )


def ablation_drop_capture(
    handle: ModelHandle,
    prompts: Sequence[str],
    sites: Sequence[HeadSite],
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    name: str = "ablation_drop_inputs",
    model_id: Optional[str] = None,
    prompt_batch: Optional[str] = None,
) -> str:
    """Measure ``metric`` on a clean pass vs a pass with ``sites`` mean-ablated,
    and record ``{baseline_metric, ablated_metric}`` as a ``model_forward``
    capture for ``compute_and_commit_metric(metric="ablation_drop", ...)``.

    Ablation = replacing each head's ``z`` with its batch-mean (a no-information
    baseline), reusing the ``head_patching`` patch machinery.
    """
    from autointerp.tools.head_patching import HeadPatch, cache_head_z, run_with_head_patches

    inputs = _tokenize(handle, prompts)
    with torch.no_grad():
        clean_logits = handle.model(**inputs, use_cache=False).logits[:, -1, :]
    baseline = float(metric(clean_logits).mean())

    layers = sorted({int(s[0]) for s in sites})
    cache = cache_head_z(handle, prompts, layers=layers)
    patches = [
        HeadPatch(
            layer=int(layer),
            head=int(head),
            source=cache[int(layer)][:, :, int(head), :].mean(dim=0, keepdim=True).expand(
                cache[int(layer)].shape[0], -1, -1
            ),
            positions=None,
        )
        for (layer, head) in sites
    ]
    ablated_logits = run_with_head_patches(handle, prompts, patches, return_logits_at=-1)
    ablated = float(metric(ablated_logits).mean())

    return record_capture(
        name,
        {"baseline_metric": baseline, "ablated_metric": ablated,
         "sites": [[int(s[0]), int(s[1])] for s in sites]},
        source="model_forward",
        model_id=model_id or getattr(handle, "model_id", None),
        prompt_batch=prompt_batch,
    )


def _tokenize(handle: ModelHandle, prompts: Sequence[str]) -> dict[str, Any]:
    from autointerp.tools.head_patching import _tokenize_batch

    return _tokenize_batch(handle, prompts)
