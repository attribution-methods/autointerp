"""Tests for the substrate-aware evaluator registry."""

from __future__ import annotations

import pytest

from autointerp.pipelines.investigation.discovery.evaluators import (
    EvaluatorNotFoundError,
    get_evaluator,
    list_evaluators,
    register,
)


def test_list_includes_components_and_features():
    keys = list_evaluators()
    assert ("components", "attn_head") in keys
    assert ("components", "mlp_neuron") in keys
    assert ("features", "sae_gemmascope") in keys


def test_attn_head_resolves():
    fn = get_evaluator(substrate="components", component_kind="attn_head")
    assert callable(fn)
    assert fn.__name__ == "evaluate"


def test_unknown_kind_rejected():
    with pytest.raises(EvaluatorNotFoundError):
        get_evaluator(substrate="components", component_kind="not_a_real_kind")


def test_components_requires_kind():
    with pytest.raises(ValueError, match="requires component_kind"):
        get_evaluator(substrate="components")


def test_features_requires_decomposition():
    with pytest.raises(ValueError, match="requires decomposition"):
        get_evaluator(substrate="features")


def test_unknown_substrate_rejected():
    with pytest.raises(ValueError, match="unknown substrate"):
        get_evaluator(substrate="nope", component_kind="attn_head")


def test_override_resolves():
    fn = get_evaluator(
        substrate="components",
        component_kind="attn_head",
        override="autointerp.pipelines.investigation.discovery.evaluators.components.attn_heads:evaluate",
    )
    assert callable(fn)


def test_override_bad_format_rejected():
    with pytest.raises(ValueError, match="module:attr"):
        get_evaluator(
            substrate="components",
            component_kind="attn_head",
            override="not_a_module_attr_form",
        )


def test_register_idempotent_same_target():
    target = "autointerp.pipelines.investigation.discovery.evaluators.components.attn_heads:evaluate"
    register("components", "attn_head", target)  # should not raise
    register("components", "attn_head", target)  # idempotent


def test_register_rejects_overwrite():
    with pytest.raises(ValueError, match="already registered"):
        register(
            "components", "attn_head", "some.other.module:evaluate"
        )


def test_stub_evaluators_raise_not_implemented():
    fn = get_evaluator(substrate="components", component_kind="mlp_neuron")
    with pytest.raises(NotImplementedError, match="mlp_neurons evaluator"):
        fn(
            None,
            model_id="x",
            device=None,
            pairs=[],
            behavioral_metric=None,
            layers=None,
            top_k=1,
            k_grid=[1],
        )

    fn = get_evaluator(substrate="features", decomposition="sae_gemmascope")
    with pytest.raises(NotImplementedError, match="SAE-feature"):
        fn(
            None,
            model_id="x",
            device=None,
            pairs=[],
            behavioral_metric=None,
            layers=None,
            top_k=1,
            k_grid=[1],
            context={"decomposition": "sae_gemmascope"},
        )
