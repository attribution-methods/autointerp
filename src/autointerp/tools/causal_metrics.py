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


def best_patch_sites(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    layers: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
    top_k: int = 3,
    min_recovery: Optional[float] = None,
) -> list[HeadSite]:
    """Sweep heads and return the top ``(layer, head)`` sites by single-head
    recovery — the candidate circuit to patch TOGETHER in
    :func:`patch_recovery_capture` / :func:`circuit_recovery_capture`.

    Most circuits are DISTRIBUTED across several heads (IOI name movers, an
    attribute spread over heads), so the single best head recovers only a
    fraction of the effect. Returns up to ``top_k`` sites ranked by recovery; if
    ``min_recovery`` is given, keeps only sites at or above it (but always at
    least the single best, so a caller never gets an empty set). ``metric`` maps
    ``[batch, vocab]`` final-token logits to ``[batch]`` (e.g. a target-minus-foil
    logit difference).
    """
    sweep = head_patch_sweep(
        handle, clean_prompts, corrupt_prompts, metric, layers=layers, heads=heads
    )
    recovery = sweep["recovery"]  # [n_layers, n_heads] tensor over chosen indices
    n_heads = recovery.shape[1]
    flat = recovery.flatten()
    order = torch.argsort(flat, descending=True)
    sites: list[HeadSite] = []
    for rank, idx in enumerate(order.tolist()):
        li, hi = idx // n_heads, idx % n_heads
        val = float(flat[idx])
        if rank == 0:
            sites.append((sweep["layers"][li], sweep["heads"][hi]))
            continue
        if len(sites) >= top_k:
            break
        if min_recovery is not None and val < min_recovery:
            break
        sites.append((sweep["layers"][li], sweep["heads"][hi]))
    return sites


def best_patch_site(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    layers: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
) -> HeadSite:
    """Single highest-recovery ``(layer, head)`` site. NOTE: most circuits are
    distributed, so one head usually recovers only a fraction — prefer
    :func:`best_patch_sites` (top-k) + :func:`circuit_recovery_capture` unless you
    have reason to believe a single head carries the effect.
    """
    return best_patch_sites(
        handle, clean_prompts, corrupt_prompts, metric, layers=layers, heads=heads, top_k=1
    )[0]


def patch_recovery_capture(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    sender: "HeadSite | Sequence[HeadSite]",
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
    logits to ``[batch]``. ``sender`` is a single ``(layer, head)`` site OR a LIST
    of sites to patch together (e.g. the top-k from :func:`best_patch_sites`) —
    use a list for distributed circuits, where one head recovers ≈ 0. For the
    common case prefer :func:`circuit_recovery_capture`, which discovers the set
    and records the capture in one call.
    """
    out = path_patch(
        handle,
        clean_prompts,
        corrupt_prompts,
        sender,
        metric,
        patch_positions=list(patch_positions) if patch_positions else None,
    )
    senders = out["senders"]
    return record_capture(
        name,
        {
            "clean_metric": float(out["clean_metric"]),
            "corrupt_metric": float(out["corrupt_metric"]),
            "patched_metric": float(out["patched_metric"]),
            "recovery": float(out["recovery"]),
            "sender": [int(senders[0][0]), int(senders[0][1])],
            "senders": [[int(s[0]), int(s[1])] for s in senders],
        },
        source="model_forward",
        model_id=model_id or getattr(handle, "model_id", None),
        prompt_batch=prompt_batch,
    )


def circuit_recovery_capture(
    handle: ModelHandle,
    clean_prompts: Sequence[str],
    corrupt_prompts: Sequence[str],
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    layers: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
    top_k: int = 3,
    min_recovery: Optional[float] = None,
    name: str = "patch_effect_recovery_inputs",
    patch_positions: Optional[Sequence[int]] = (-1,),
    model_id: Optional[str] = None,
    prompt_batch: Optional[str] = None,
) -> str:
    """One-call push-button ``patch_effect_recovery`` for a DISTRIBUTED circuit.

    Sweeps heads, takes the top ``top_k`` by single-head recovery
    (:func:`best_patch_sites`), patches that whole set together
    (:func:`path_patch`), and records the gate-passing ``model_forward`` capture.
    This is the right default for a circuit question: patching a single head
    almost always reports recovery ≈ 0 even when the circuit is real, which reads
    as a false negative. Returns the capture relpath for
    ``compute_and_commit_metric(metric="patch_effect_recovery", ...)``.
    """
    sites = best_patch_sites(
        handle, clean_prompts, corrupt_prompts, metric,
        layers=layers, heads=heads, top_k=top_k, min_recovery=min_recovery,
    )
    return patch_recovery_capture(
        handle, clean_prompts, corrupt_prompts, sites, metric,
        name=name, patch_positions=patch_positions,
        model_id=model_id, prompt_batch=prompt_batch,
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
