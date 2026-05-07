"""Phenomenon-agnostic protocol for benchmark modules.

A *benchmark* knows two things about a phenomenon:

1. **How to generate paired stimuli** — given a spec, produce a list of
   ``StimulusPair`` objects (clean / corrupted prompt + the target / foil
   token ids the model is supposed to choose between).
2. **How to score behavior at a position** — given the model's logits at
   the prediction position and the pair, return a scalar that says "how
   well did the model do this trial."

The discovery harness's substrate evaluators (head ablation, MLP-neuron
ablation, SAE-feature steering, …) call into this protocol so they stay
phenomenon-agnostic. Adding IOI / induction / sycophancy / refusal /
factual recall is one new module under ``src/autointerp/benchmarks/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class StimulusPair:
    """One paired-contrast trial.

    Fields are intentionally minimal — substrate-specific evaluators
    should rely only on these. Anything richer (per-token positions of
    interest, sub-token role labels, cross-pair relationships) goes in
    ``metadata``.
    """

    pair_id: str
    clean_prompt: str
    corrupted_prompt: str
    target_token_id: int
    foil_token_id: int
    # Index of the prediction position in the *tokenized* clean prompt.
    # Defaults to -1 (the last position), which is the right answer for
    # next-token classification phenomena like IOI.
    prediction_position: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


# A behavioral metric reduces a logits tensor at the prediction position to a
# scalar. ``logits`` has shape ``[vocab]`` (single trial) or ``[batch, vocab]``
# (batched trials); the metric returns a Python float for the former and a
# ``[batch]`` tensor / float-list for the latter. By convention, **higher is
# better** for the clean (target) behavior — evaluators rely on this sign
# convention to compute normalized recovery curves.
BehavioralMetric = Callable[[Any, StimulusPair], float]


class Benchmark(Protocol):
    """The shape every benchmark module must implement.

    This is a structural protocol: no inheritance required. Any module
    exposing module-level ``generate_pairs`` and ``behavioral_metric``
    callables works.
    """

    GENERATOR_ID: str

    def generate_pairs(self, spec: Any, *, n_pairs: int, seed: int) -> list[StimulusPair]:
        """Produce paired stimuli from a spec's dataset / contrast fields.

        Implementations should be deterministic given ``seed``; they should
        respect ``spec.dataset.metadata`` knobs (single-token name lists,
        templates, etc.) when present, but ship sensible defaults so that
        a minimal spec still produces working pairs.
        """
        ...

    def behavioral_metric(self, logits: Any, pair: StimulusPair) -> float:
        """Score a single trial's logits.

        Convention: return ``logit(target) - logit(foil)`` (higher = clean
        behavior). Substrate evaluators normalize *across* the K-sweep,
        not against an absolute scale, so the only constraint is that the
        sign points at the clean direction.
        """
        ...


__all__ = [
    "BehavioralMetric",
    "Benchmark",
    "StimulusPair",
]
