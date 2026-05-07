"""Benchmark registry — keyed on ``DatasetSpec.generator_id``.

The discovery harness uses :func:`get_benchmark` to resolve a spec's
``dataset.generator_id`` to a benchmark module exposing
``generate_pairs`` + ``behavioral_metric``. Adding a new phenomenon means
dropping one module under ``src/autointerp/benchmarks/`` and registering
its ``GENERATOR_ID`` here (or having the module self-register at import
time via :func:`register`).
"""

from __future__ import annotations

import importlib
from typing import Any

# Module-level mapping. Keep keys sorted alphabetically when extending so
# diffs stay clean.
_REGISTRY: dict[str, str] = {
    # generator_id -> dotted module path (lazily imported)
    "ioi_template_generator": "autointerp.benchmarks.ioi",
}


class BenchmarkNotFoundError(KeyError):
    """Raised when a generator_id has no registered benchmark module."""


def register(generator_id: str, module_path: str) -> None:
    """Register a benchmark module by ``generator_id``.

    Idempotent: re-registering the same ``(generator_id, module_path)``
    pair is a no-op, but pointing the same id at a different module is an
    error (suggests a typo or a duplicate plug-in).
    """
    existing = _REGISTRY.get(generator_id)
    if existing is not None and existing != module_path:
        raise ValueError(
            f"benchmark `{generator_id}` is already registered to "
            f"{existing!r}; refusing to overwrite with {module_path!r}"
        )
    _REGISTRY[generator_id] = module_path


def get_benchmark(generator_id: str) -> Any:
    """Import and return the benchmark module for ``generator_id``."""
    module_path = _REGISTRY.get(generator_id)
    if module_path is None:
        raise BenchmarkNotFoundError(
            f"No benchmark registered for generator_id={generator_id!r}. "
            f"Known ids: {sorted(_REGISTRY)}"
        )
    return importlib.import_module(module_path)


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "BenchmarkNotFoundError",
    "get_benchmark",
    "list_benchmarks",
    "register",
]
