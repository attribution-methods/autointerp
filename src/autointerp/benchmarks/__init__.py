"""Phenomenon-agnostic benchmark layer.

Each benchmark module exports a ``GENERATOR_ID`` that matches the spec's
``DatasetSpec.generator_id``, plus ``generate_pairs(spec, ...)`` and
``behavioral_metric(logits, pair) -> float``. The discovery harness's
substrate evaluators consume these so they stay phenomenon-agnostic.

Add a benchmark by:

1. Creating ``src/autointerp/benchmarks/<id>.py`` with the protocol
   members.
2. Registering it in ``_registry.py`` (``_REGISTRY[<id>] =
   "autointerp.benchmarks.<id>"``).
"""

from ._protocol import BehavioralMetric, Benchmark, StimulusPair  # noqa: F401
from ._registry import (  # noqa: F401
    BenchmarkNotFoundError,
    get_benchmark,
    list_benchmarks,
    register,
)

__all__ = [
    "BehavioralMetric",
    "Benchmark",
    "BenchmarkNotFoundError",
    "StimulusPair",
    "get_benchmark",
    "list_benchmarks",
    "register",
]
