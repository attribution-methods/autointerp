"""Substrate-aware evaluator registry for the discovery sub-agent.

An *evaluator* is the harness-side ML callable that loads the model once,
runs a candidate algorithm's ``score(...)``, and returns a payload of
metric values + ranked top features. The discovery harness picks the
evaluator based on the spec's ``DiscoveryConfig`` (``substrate``,
``component_kinds``, ``decomposition``).

The registry shape is:

- For ``substrate="components"``: keyed by ``component_kind`` (e.g.
  ``"attn_head"`` → ``components.attn_heads.evaluate``). When a stage
  declares multiple kinds, the harness picks the first registered
  evaluator that supports that combination (or, in the simple
  single-kind case, just that one evaluator).
- For ``substrate="features"``: keyed by ``decomposition`` (e.g.
  ``"sae_gemmascope"`` → ``features.sae.evaluate`` with appropriate
  loader wiring).

An explicit ``DiscoveryConfig.evaluator = "module:attr"`` always wins
over the registry lookup.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

# ``(substrate, kind_or_decomposition) -> dotted module:attr``. Stubs for
# everything except attn_head; adding a real evaluator is one new module
# under ``components/`` or ``features/`` plus one entry here.
_REGISTRY: dict[tuple[str, str], str] = {
    # components
    ("components", "attn_head"): (
        "autointerp.pipelines.investigation.discovery.evaluators.components.attn_heads:evaluate"
    ),
    ("components", "mlp_neuron"): (
        "autointerp.pipelines.investigation.discovery.evaluators.components.mlp_neurons:evaluate"
    ),
    ("components", "mlp_layer"): (
        "autointerp.pipelines.investigation.discovery.evaluators.components.mlp_layers:evaluate"
    ),
    ("components", "residual_layer"): (
        "autointerp.pipelines.investigation.discovery.evaluators.components.residual_layers:evaluate"
    ),
    # features — namespaced by decomposition family
    ("features", "sae_gemmascope"): (
        "autointerp.pipelines.investigation.discovery.evaluators.features.sae:evaluate"
    ),
    ("features", "sae_bloom_gpt2_small"): (
        "autointerp.pipelines.investigation.discovery.evaluators.features.sae:evaluate"
    ),
    ("features", "probe_v1"): (
        "autointerp.pipelines.investigation.discovery.evaluators.features.probes:evaluate"
    ),
}


class EvaluatorNotFoundError(KeyError):
    """Raised when (substrate, kind/decomposition) has no registered evaluator."""


def _resolve(spec: str) -> Callable[..., Any]:
    """Resolve a ``module:attr`` string to a callable. Raises if the import or
    attribute lookup fails — the caller is expected to surface the failure
    as a tool error so the master agent sees a useful message."""
    if ":" not in spec:
        raise ValueError(
            f"Evaluator spec must be `module:attr`, got {spec!r}"
        )
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise EvaluatorNotFoundError(
            f"Module {module_path!r} has no attribute {attr!r}"
        )
    return getattr(module, attr)


def register(
    substrate: str,
    kind_or_decomposition: str,
    module_attr: str,
) -> None:
    """Register an evaluator for ``(substrate, kind_or_decomposition)``."""
    key = (substrate, kind_or_decomposition)
    existing = _REGISTRY.get(key)
    if existing is not None and existing != module_attr:
        raise ValueError(
            f"evaluator {key} already registered to {existing!r}; "
            f"refusing to overwrite with {module_attr!r}"
        )
    _REGISTRY[key] = module_attr


def get_evaluator(
    *,
    substrate: str,
    component_kind: str | None = None,
    decomposition: str | None = None,
    override: str | None = None,
) -> Callable[..., Any]:
    """Resolve the evaluator callable for a discovery configuration.

    Parameters
    ----------
    substrate:
        ``"components"`` or ``"features"``.
    component_kind:
        Required when ``substrate == "components"`` (e.g. ``"attn_head"``).
    decomposition:
        Required when ``substrate == "features"`` (e.g. ``"sae_gemmascope"``).
    override:
        Optional ``module:attr``. If set, returned directly without a
        registry lookup. Used for project-local evaluators.
    """
    if override is not None:
        return _resolve(override)

    if substrate == "components":
        if component_kind is None:
            raise ValueError(
                "get_evaluator: substrate='components' requires component_kind"
            )
        key = ("components", component_kind)
    elif substrate == "features":
        if decomposition is None:
            raise ValueError(
                "get_evaluator: substrate='features' requires decomposition"
            )
        key = ("features", decomposition)
    else:
        raise ValueError(
            f"get_evaluator: unknown substrate {substrate!r}. "
            f"Expected 'components' or 'features'."
        )

    spec = _REGISTRY.get(key)
    if spec is None:
        raise EvaluatorNotFoundError(
            f"No evaluator registered for {key}. "
            f"Known: {sorted(_REGISTRY.keys())}"
        )
    return _resolve(spec)


def list_evaluators() -> list[tuple[str, str]]:
    """Return ``[(substrate, kind_or_decomposition), ...]``, sorted."""
    return sorted(_REGISTRY)


__all__ = [
    "EvaluatorNotFoundError",
    "get_evaluator",
    "list_evaluators",
    "register",
]
