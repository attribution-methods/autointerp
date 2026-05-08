"""Generalized candidate dataclass for the discovery sub-agent.

A :class:`Candidate` represents one causally targetable site, regardless of
substrate (attention head, MLP neuron, MLP layer, residual-layer site, SAE
feature, transcoder feature, probe direction, …). The sub-agent's
``score(...)`` returns a ranked list of these; the harness's evaluator
performs substrate-appropriate interventions.

The dataclass shape is intentionally substrate-agnostic — the ``kind`` field
discriminates within a candidate list. This lets a single algorithm propose
mixed-kind candidates when that's meaningful (e.g. attn heads and MLP
neurons in the same ranking), and lets the harness validate that an
algorithm running under ``substrate=components`` does not return
``kind="sae_feature"`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# The set of recognized component kinds. Keep in sync with
# ``DiscoveryConfig.component_kinds`` in ``autointerp.spec``.
COMPONENT_KINDS: tuple[str, ...] = (
    "attn_head",
    "mlp_layer",
    "mlp_neuron",
    "residual_layer",
)

# Feature substrates are namespaced by the decomposition that owns them.
# Common values: "sae_feature", "transcoder_feature", "probe", "pca".
FEATURE_KINDS: tuple[str, ...] = (
    "sae_feature",
    "transcoder_feature",
    "probe",
    "pca",
)


CandidateKind = Literal[
    "attn_head", "mlp_layer", "mlp_neuron", "residual_layer",
    "sae_feature", "transcoder_feature", "probe", "pca",
]


@dataclass(frozen=True)
class Candidate:
    """One causally targetable site.

    Attributes
    ----------
    layer:
        Layer index where the site lives.
    idx:
        Per-layer integer key. Concrete meaning depends on ``kind``:
        head index for ``attn_head``; neuron index for ``mlp_neuron``;
        feature index for ``sae_feature`` / ``transcoder_feature`` /
        ``probe``; ignored (set to 0) for ``mlp_layer`` and
        ``residual_layer``.
    score:
        Signed importance. Convention: positive = promotes the *clean*
        (target) behavior; magnitude is the ranking signal.
    kind:
        Discriminator over substrates. See ``COMPONENT_KINDS`` and
        ``FEATURE_KINDS`` for the recognized values.
    token_pos:
        Optional token position the score is specific to. ``None`` =
        all-positions (the harness applies the intervention everywhere).
    metadata:
        Free-form per-candidate dict (attribution method name, gradient
        norm, attention pattern stats, …). Surfaces in the harness summary
        and the leaderboard.
    """

    layer: int
    idx: int
    score: float
    kind: str = "attn_head"
    token_pos: int | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_component(self) -> bool:
        return self.kind in COMPONENT_KINDS

    @property
    def is_feature(self) -> bool:
        return self.kind in FEATURE_KINDS

    def address(self) -> str:
        """Stable string id for logging / leaderboard / experiment records."""
        token_part = "" if self.token_pos is None else f"@t{self.token_pos}"
        return f"L{self.layer}.{self.kind}.{self.idx}{token_part}"


# Backwards-compatibility alias for code paths that imported the old name.
# The pre-generalization API exposed ``FeatureCandidate(layer, feature_id,
# score, token_pos, metadata)``; ``Candidate`` is a strict superset (adds
# ``kind`` with a sensible default). Existing call sites that construct
# with positional args keep working because ``layer`` and ``idx`` (was
# ``feature_id``) are still positions 0/1.
FeatureCandidate = Candidate


__all__ = [
    "COMPONENT_KINDS",
    "FEATURE_KINDS",
    "Candidate",
    "CandidateKind",
    "FeatureCandidate",
]
