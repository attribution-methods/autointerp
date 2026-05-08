"""Template for ``algorithm_v{N}.py`` discovery candidates.

This module is the *generic* fallback: it ships a deterministic stub that
keeps the harness's ``--dry-run`` path round-tripping. For real runs the
discovery loop substitutes the substrate-specific template under
``discovery/templates/`` (``components_baseline.py`` for attention heads /
MLP neurons / etc.; ``features_baseline.py`` for SAE features / probes).

Copy whichever template is appropriate for your substrate into your
session directory as ``algorithm_v{N}.py`` and edit ``score(...)`` to
implement your attribution method. Return ``list[Candidate]`` sorted by
descending ``abs(score)``, length ``<= top_k``.
"""

from __future__ import annotations

from typing import Any

from autointerp.pipelines.investigation.discovery.candidate import Candidate

NAME = "algorithm_template"


def score(
    pairs: list,
    *,
    loader: Any,
    layers: list[int],
    device: str,
    top_k: int,
    context: Any = None,
) -> list[Candidate]:
    """Compute importance scores from raw model components and stimulus pairs.

    Replace the body with your attribution method. The default implementation
    is a deterministic stub: it returns ``top_k`` no-op candidates so the
    harness round-trips end-to-end during ``--dry-run``.

    Parameters
    ----------
    pairs:
        Iterable of stimulus pairs from the spec's benchmark. Each entry
        carries a clean prompt, a corrupted prompt, and the target / foil
        token ids the behavioral metric uses.
    loader:
        Model bundle (``DiscoveryLoader``-shaped — at minimum ``.model``
        and ``.tokenizer``; SAE-substrate runs also expose ``.saes``).
    layers:
        Layer indices to score over.
    device:
        Torch device string (``"cuda:0"``, ``"cpu"``).
    top_k:
        Maximum number of candidates to return.
    context:
        Free-form dict / dataclass with discovery context (substrate,
        component_kinds, k_grid, behavior id, …).

    Returns
    -------
    list[Candidate], length <= ``top_k``, sorted by descending
    ``abs(score)``.
    """
    # Generic dry-run baseline: deterministic placeholder candidates.
    # Substrate-specific templates ship real attribution methods; this
    # one only exists so the harness's ``--dry-run`` exercises end to end
    # without needing any model loaded.
    candidates: list[Candidate] = []
    for i, layer in enumerate(layers[: max(top_k, 1)]):
        candidates.append(
            Candidate(
                layer=int(layer),
                idx=i,
                score=1.0 / (i + 1),
                kind="attn_head",
                token_pos=None,
                metadata={"reason": "generic dry-run stub"},
            )
        )
    return candidates[:top_k]
