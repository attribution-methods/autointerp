"""Regression tests for component/layer addressing in the patching tools.

A real run died here: the driver called
``patch_generation(handle, prompts, layer, component, site)`` — misordering the
positional args so the component name ("self_attn") landed in the LAYER slot.
The old error, ``Invalid component spec: self_attn``, never revealed the
arg-order mistake, so the weak driver spiraled and bailed to a needless spec
revision (confabulating that per-head "L6H0" addressing was unsupported — it is
NOT: ``L6H0`` parses fine). These tests pin the actionable errors + aliases that
let it self-correct instead.
"""

from __future__ import annotations

import pytest

# autointerp.tools.activations imports torch at module level — an extra dep CI's
# core install intentionally lacks. Skip there; run wherever torch exists.
pytest.importorskip("torch")

from autointerp.tools.activations import (  # noqa: E402
    _COMPONENT_CANON,
    component_module,
    parse_component_spec,
)


def test_component_name_in_layer_slot_gives_argorder_hint() -> None:
    # The exact field failure: a component name where a layer is expected.
    for bad in ("self_attn", "attn", "head", "mlp", "resid"):
        with pytest.raises(ValueError) as exc:
            parse_component_spec(bad, 12)
        msg = str(exc.value)
        assert "COMPONENT" in msg and "argument order" in msg, msg


def test_layer_specs_including_per_head_parse() -> None:
    # The format the agent claimed was unsupported actually works.
    s = parse_component_spec("L6H0", 12)
    assert s.layers == [6] and s.component == "head" and s.heads == [0]
    assert parse_component_spec("L6MLP", 12).component == "mlp"
    assert parse_component_spec("L6ATTN", 12).component == "attn"
    assert parse_component_spec("L6-8", 12).layers == [6, 7, 8]
    assert parse_component_spec("L3", 12).layers == [3]


class _FakeLayer:
    self_attn = "ATTN_MODULE"
    mlp = "MLP_MODULE"


class _FakeHandle:
    def layer(self, _idx: int) -> _FakeLayer:
        return _FakeLayer()


def test_component_module_normalizes_aliases() -> None:
    h = _FakeHandle()
    # the alias the agent used now resolves instead of erroring
    assert component_module(h, 6, "self_attn") == "ATTN_MODULE"
    assert component_module(h, 6, "attention") == "ATTN_MODULE"
    assert component_module(h, 6, "ffn") == "MLP_MODULE"
    assert isinstance(component_module(h, 6, "resid"), _FakeLayer)  # whole block
    # the canon map is the single source of truth
    assert _COMPONENT_CANON["self_attn"] == "attn"
    assert _COMPONENT_CANON["residual"] == "resid"


def test_component_module_unknown_lists_valid_options() -> None:
    with pytest.raises(ValueError) as exc:
        component_module(_FakeHandle(), 6, "L6H0")  # a layer spec, not a component
    msg = str(exc.value)
    assert "resid" in msg and "mlp" in msg and "attn" in msg
