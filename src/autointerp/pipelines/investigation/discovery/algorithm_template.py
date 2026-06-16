"""Template for ``algorithm_v{N}.py`` discovery candidates.

The discovery loop seeds a session with this file and asks the sub-agent to
rewrite ``score(...)`` each iteration. The default body is a deterministic
stub so the harness's ``--dry-run`` path round-trips end-to-end without a
model loaded (used by tests and CI).

Contract for ``score``:

- return ``list[Candidate]`` sorted by descending ``abs(score)``;
- length ``<= top_k``;
- pure ranking — no global state, no I/O. The harness owns model loading,
  ablation, steering, and metric computation.
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
    """Rank candidate sites for a behavior from contrast pairs.

    Replace the body with a real attribution method (e.g. activation
    patching deltas, gradient-times-activation, SAE feature attribution).

    Parameters
    ----------
    pairs:
        Contrast pairs (clean / corrupted prompts + target/foil tokens).
    loader:
        Model bundle (at minimum ``.model`` / ``.tokenizer``; SAE runs also
        expose ``.saes``). ``None`` under ``--dry-run``.
    layers:
        Layer indices to score over.
    device:
        Torch device string.
    top_k:
        Maximum number of candidates to return.
    context:
        Free-form dict with discovery context (behavior id, metadata, ...).

    Returns
    -------
    list[Candidate], length <= ``top_k``, sorted by descending ``abs(score)``.
    """
    # Deterministic dry-run baseline: monotonically decreasing placeholder
    # scores so the harness can synthesize a well-formed AUC curve without a
    # model. A real candidate computes these from interventions on `loader`.
    candidates: list[Candidate] = []
    for i, layer in enumerate(layers[: max(top_k, 1)]):
        candidates.append(
            Candidate(
                layer=int(layer),
                idx=i,
                score=1.0 / (i + 1),
                kind="attn_head",
                token_pos=None,
                metadata={"reason": "dry-run stub"},
            )
        )
    return candidates[:top_k]
