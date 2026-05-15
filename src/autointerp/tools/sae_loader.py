"""Pretrained SAE loader.

Thin wrapper around ``sae_lens.SAE.from_pretrained`` that gives the agent
a single entry point for Gemma Scope, Llama Scope, and JumpReLU SAEs.
The heavy weight load only happens when the agent actually calls
``load_pretrained_sae``; tests mock the wrapper.

The companion ``sae_labels`` module handles Neuronpedia label lookups;
``autointerp.tools.sae`` (existing) handles post-processing of feature
activations once an SAE is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# A small registry of common (release, sae_id) pairs the agent can pick
# from when it doesn't already know the canonical id. NOT exhaustive —
# sae-lens has ~thousands of releases. The skill card points the agent at
# the full registry; this is just an ergonomic shortlist.
KNOWN_RELEASES: dict[str, dict[str, str]] = {
    "gemma-scope-2b-pt-res-canonical": {
        "model": "google/gemma-2-2b",
        "default_sae_id": "layer_12/width_16k/canonical",
        "labels_release": "gemma-2-2b",
    },
    "gemma-scope-9b-pt-res-canonical": {
        "model": "google/gemma-2-9b",
        "default_sae_id": "layer_20/width_131k/canonical",
        "labels_release": "gemma-2-9b",
    },
    "gemma-scope-9b-it-res-canonical": {
        "model": "google/gemma-2-9b-it",
        "default_sae_id": "layer_20/width_131k/canonical",
        "labels_release": "gemma-2-9b-it",
    },
    "llama_scope_lxr_8x": {
        # Base Llama-3.1-8B, residual stream, 8x width. Use on the
        # instruct variant (Llama-3.1-8B-Instruct) too — base→instruct
        # SAE transfer works at slightly higher reconstruction loss.
        "model": "meta-llama/Llama-3.1-8B",
        "default_sae_id": "l19r_8x",
        "labels_release": "llama-3.1-8b",
    },
    "goodfire-llama-3.1-8b-instruct": {
        # Trained directly on Llama-3.1-8B-Instruct — best fidelity for
        # the instruct model. Single mid-late layer release.
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "default_sae_id": "layer_19",
        "labels_release": "llama-3.1-8b-instruct",
    },
    "llama-3.1-8b-instruct-andyrdt": {
        # Trained on Llama-3.1-8B-Instruct, multiple layers + trainers.
        # Pick a layer near the late-middle of the residual stream.
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "default_sae_id": "resid_post_layer_19_trainer_1",
        "labels_release": "llama-3.1-8b-instruct",
    },
    "llama_scope_r1_distill": {
        # SAEs trained on DeepSeek-R1-Distill-Llama-8B — NOT vanilla
        # Llama. Use only when targeting the R1-distilled model.
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "default_sae_id": "l19r_800m_slimpajama",
        "labels_release": "deepseek-r1-distill-llama-8b",
    },
}


@dataclass(frozen=True)
class SAEHandle:
    """Resolved SAE + the metadata callers need to look up labels.

    ``sae`` is the ``sae_lens.SAE`` object; use it directly for ``encode``
    / ``decode`` / ``W_dec`` access. ``release`` and ``sae_id`` are the
    sae-lens identifiers; ``labels_release`` is the Neuronpedia release
    string (often slightly different — pass to
    ``sae_labels.lookup_neuronpedia_label``).
    """

    sae: Any
    release: str
    sae_id: str
    labels_release: str
    layer: int
    n_features: int
    component: str  # "resid_post" | "attn_out" | "mlp_out" | "transcoder" | ...


def load_pretrained_sae(
    release: str,
    *,
    sae_id: str | None = None,
    device: str = "cuda",
) -> SAEHandle:
    """Load a pretrained SAE by ``(release, sae_id)``.

    ``release`` is a sae-lens release string (e.g. ``gemma-scope-9b-pt-res-canonical``).
    ``sae_id`` is the per-release SAE identifier (e.g. ``layer_20/width_131k/canonical``);
    omit to use the default in ``KNOWN_RELEASES``, or look up the full set
    in the sae-lens registry.

    Raises ``RuntimeError`` if sae-lens isn't installed (the
    ``[mechinterp]`` extra ships it).
    """
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise RuntimeError(
            "sae-lens is required for load_pretrained_sae. "
            "Install with `pip install -e .[mechinterp]`."
        ) from exc

    info = KNOWN_RELEASES.get(release)
    if sae_id is None:
        if info is None:
            raise ValueError(
                f"Unknown release {release!r} and no sae_id passed. "
                "Either pass sae_id explicitly or pick from "
                f"{sorted(KNOWN_RELEASES)}."
            )
        sae_id = info["default_sae_id"]

    sae, _cfg_dict, _sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device
    )

    layer = int(getattr(sae.cfg, "hook_layer", -1))
    component = str(getattr(sae.cfg, "hook_name", "unknown"))
    d_sae = getattr(sae.cfg, "d_sae", None)
    if d_sae is None:
        d_sae = sae.W_dec.shape[0]
    n_features = int(d_sae)
    labels_release = (
        info["labels_release"] if info is not None else release.split("-canonical")[0]
    )

    return SAEHandle(
        sae=sae,
        release=release,
        sae_id=sae_id,
        labels_release=labels_release,
        layer=layer,
        n_features=n_features,
        component=component,
    )


__all__ = ["KNOWN_RELEASES", "SAEHandle", "load_pretrained_sae"]
