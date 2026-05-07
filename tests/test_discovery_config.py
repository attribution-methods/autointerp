"""Validator tests for the new ``DiscoveryConfig`` and its interaction
with ``StageSpec``."""

from __future__ import annotations

import pytest

from autointerp.spec import (
    DiscoveryConfig,
    InvestigationStage,
    MetricName,
    PatternId,
    StageSpec,
    ToolName,
)


# ---------------------------------------------------------------------------
# DiscoveryConfig field-level
# ---------------------------------------------------------------------------


def test_components_default_kinds():
    cfg = DiscoveryConfig(substrate="components")
    assert cfg.component_kinds == ["attn_head"]


def test_components_explicit_kinds():
    cfg = DiscoveryConfig(
        substrate="components", component_kinds=["attn_head", "mlp_neuron"]
    )
    assert "attn_head" in cfg.component_kinds


def test_components_rejects_decomposition():
    with pytest.raises(Exception, match="must not set decomposition"):
        DiscoveryConfig(substrate="components", decomposition="sae_x")


def test_features_requires_decomposition():
    with pytest.raises(Exception, match="requires decomposition"):
        DiscoveryConfig(substrate="features")


def test_features_with_decomposition_ok():
    cfg = DiscoveryConfig(
        substrate="features",
        decomposition="sae_gemmascope",
        decomposition_layers=[12, 20],
    )
    assert cfg.substrate == "features"
    assert cfg.decomposition == "sae_gemmascope"


def test_k_grid_must_be_sorted_ascending():
    with pytest.raises(Exception, match="sorted ascending"):
        DiscoveryConfig(substrate="components", k_grid=[5, 1, 10])


def test_k_grid_must_be_positive():
    with pytest.raises(Exception, match="positive ints"):
        DiscoveryConfig(substrate="components", k_grid=[0, 5, 10])


def test_components_rejects_empty_kinds():
    with pytest.raises(Exception, match="component_kinds"):
        DiscoveryConfig(substrate="components", component_kinds=[])


# ---------------------------------------------------------------------------
# StageSpec interaction
# ---------------------------------------------------------------------------


def _stage(*, with_tool: bool, with_disco: bool) -> StageSpec:
    tools = [ToolName.DISCOVER_FEATURES] if with_tool else [ToolName.BLACKBOX_PROBE]
    disco = (
        DiscoveryConfig(substrate="components", component_kinds=["attn_head"])
        if with_disco
        else None
    )
    return StageSpec(
        stage=InvestigationStage.FEATURE_DISCOVERY if with_tool else InvestigationStage.BLACK_BOX,
        pattern=PatternId.BLACKBOX_THEN_PATCHING,
        tools=tools,
        metrics=[MetricName.PATCH_EFFECT_RECOVERY],
        discovery=disco,
    )


def test_discover_features_tool_requires_discovery_config():
    with pytest.raises(Exception, match="`discovery` is not set"):
        _stage(with_tool=True, with_disco=False)


def test_discovery_config_requires_discover_features_tool():
    with pytest.raises(Exception, match="does not list `discover_features`"):
        _stage(with_tool=False, with_disco=True)


def test_stage_with_tool_and_config_ok():
    s = _stage(with_tool=True, with_disco=True)
    assert s.discovery is not None
    assert s.discovery.substrate == "components"


def test_stage_without_tool_or_config_ok():
    s = _stage(with_tool=False, with_disco=False)
    assert s.discovery is None
