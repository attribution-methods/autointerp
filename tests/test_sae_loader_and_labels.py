"""Tests for the SAE loader and Neuronpedia label-lookup wrappers.

These never hit the network or load real SAE weights — sae-lens and
``requests`` are both monkeypatched. The point is to lock in the wrapper
contract (KNOWN_RELEASES shape, default sae_id resolution, label parsing,
caching) without GPU or external deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

# autointerp.tools imports torch at package level — a mechinterp-extra
# dependency that CI (core install) intentionally lacks. Skip there; run
# wherever torch exists.
pytest.importorskip("torch")

from autointerp.tools import sae_labels, sae_loader  # noqa: E402

# ---- sae_loader -------------------------------------------------------------


@dataclass
class _FakeCfg:
    hook_layer: int = 20
    hook_name: str = "blocks.20.hook_resid_post"
    d_sae: int = 131072


@dataclass
class _FakeSAE:
    cfg: _FakeCfg
    W_dec: object = None  # not exercised in these tests


class _FakeSAEModule:
    """Stand-in for ``sae_lens.SAE`` exposing a ``from_pretrained`` classmethod."""

    captured: dict = {}

    @classmethod
    def from_pretrained(cls, *, release: str, sae_id: str, device: str):
        cls.captured = {"release": release, "sae_id": sae_id, "device": device}
        return _FakeSAE(cfg=_FakeCfg()), {"cfg": "ignored"}, None


def _patch_sae_lens(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake_module = types.ModuleType("sae_lens")
    fake_module.SAE = _FakeSAEModule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sae_lens", fake_module)


def test_load_pretrained_sae_uses_default_sae_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sae_lens(monkeypatch)
    handle = sae_loader.load_pretrained_sae("gemma-scope-9b-pt-res-canonical", device="cpu")
    assert _FakeSAEModule.captured["sae_id"] == "layer_20/width_131k/canonical"
    assert handle.release == "gemma-scope-9b-pt-res-canonical"
    assert handle.labels_release == "gemma-2-9b"
    assert handle.layer == 20
    assert handle.n_features == 131072


def test_load_pretrained_sae_explicit_sae_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sae_lens(monkeypatch)
    handle = sae_loader.load_pretrained_sae(
        "gemma-scope-9b-pt-res-canonical",
        sae_id="layer_25/width_16k/canonical",
        device="cpu",
    )
    assert _FakeSAEModule.captured["sae_id"] == "layer_25/width_16k/canonical"
    assert handle.sae_id == "layer_25/width_16k/canonical"


def test_load_pretrained_sae_unknown_release_requires_explicit_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sae_lens(monkeypatch)
    with pytest.raises(ValueError, match="Unknown release"):
        sae_loader.load_pretrained_sae("not-a-known-release", device="cpu")


def test_load_pretrained_sae_missing_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without sae-lens installed, the wrapper raises a helpful error."""
    import sys

    monkeypatch.setitem(sys.modules, "sae_lens", None)
    with pytest.raises(RuntimeError, match="sae-lens is required"):
        sae_loader.load_pretrained_sae(
            "gemma-scope-9b-pt-res-canonical", device="cpu"
        )


def test_known_releases_shape() -> None:
    for release, info in sae_loader.KNOWN_RELEASES.items():
        assert "default_sae_id" in info
        assert "labels_release" in info
        assert "model" in info
        assert release.replace("-", "").isalnum() or "/" not in release


# ---- sae_labels -------------------------------------------------------------


def _patch_http(monkeypatch: pytest.MonkeyPatch, payload: object) -> dict:
    captured: dict = {}

    def fake_get(url: str, *, timeout: float = 15) -> object:
        captured["url"] = url
        captured["timeout"] = timeout
        return payload

    monkeypatch.setattr(sae_labels, "_http_get", fake_get)
    return captured


def _isolate_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOINTERP_NEURONPEDIA_CACHE", str(tmp_path / "np_cache"))


def test_lookup_label_basic(monkeypatch, tmp_path: Path) -> None:
    _isolate_cache(monkeypatch, tmp_path)
    payload = {"explanations": [{"description": "expressions of triumph or joy"}]}
    captured = _patch_http(monkeypatch, payload)
    label = sae_labels.lookup_neuronpedia_label("gemma-2-9b", 20, 14729)
    assert label == "expressions of triumph or joy"
    assert "feature/gemma-2-9b/20-res/14729" in captured["url"]


def test_lookup_label_handles_missing_explanations(monkeypatch, tmp_path: Path) -> None:
    _isolate_cache(monkeypatch, tmp_path)
    _patch_http(monkeypatch, {"description": "fallback path"})
    assert sae_labels.lookup_neuronpedia_label("gemma-2-9b", 20, 1) == "fallback path"


def test_lookup_label_returns_none_on_empty(monkeypatch, tmp_path: Path) -> None:
    _isolate_cache(monkeypatch, tmp_path)
    _patch_http(monkeypatch, {"explanations": []})
    assert sae_labels.lookup_neuronpedia_label("gemma-2-9b", 20, 1) is None


def test_lookup_label_cached(monkeypatch, tmp_path: Path) -> None:
    """Second call reads the on-disk cache, not the network."""
    _isolate_cache(monkeypatch, tmp_path)
    payload = {"explanations": [{"description": "first"}]}
    _patch_http(monkeypatch, payload)
    first = sae_labels.lookup_neuronpedia_label("gemma-2-9b", 20, 7)
    assert first == "first"

    # Now make any network call raise — the cache should serve.
    def boom(*_a, **_k):
        raise AssertionError("HTTP should not be re-invoked when cached")

    monkeypatch.setattr(sae_labels, "_http_get", boom)
    second = sae_labels.lookup_neuronpedia_label("gemma-2-9b", 20, 7)
    assert second == "first"


def test_lookup_label_swallows_http_errors(monkeypatch, tmp_path: Path) -> None:
    _isolate_cache(monkeypatch, tmp_path)

    def fail(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(sae_labels, "_http_get", fail)
    assert sae_labels.lookup_neuronpedia_label("gemma-2-9b", 20, 1) is None


def test_search_returns_ranked_features(monkeypatch, tmp_path: Path) -> None:
    _isolate_cache(monkeypatch, tmp_path)
    payload = [
        {"index": 14729, "description": "triumph", "score": 0.92},
        {"index": 88421, "description": "satisfied", "score": 0.81},
    ]
    captured = _patch_http(monkeypatch, payload)
    rows = sae_labels.search_features_by_label("gemma-2-9b", 20, "joy triumph", k=5)
    assert [r.feature_id for r in rows] == [14729, 88421]
    assert rows[0].score > rows[1].score
    assert "query=joy triumph" in captured["url"]
    assert "maxResults=5" in captured["url"]


def test_search_handles_results_envelope(monkeypatch, tmp_path: Path) -> None:
    _isolate_cache(monkeypatch, tmp_path)
    _patch_http(monkeypatch, {"results": [{"index": 1, "label": "x", "similarity": 0.5}]})
    rows = sae_labels.search_features_by_label("gemma-2-9b", 20, "x", k=1)
    assert rows == [sae_labels.FeatureLabel(feature_id=1, label="x", score=0.5)]


def test_search_returns_empty_on_failure(monkeypatch, tmp_path: Path) -> None:
    _isolate_cache(monkeypatch, tmp_path)

    def fail(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(sae_labels, "_http_get", fail)
    assert sae_labels.search_features_by_label("gemma-2-9b", 20, "x") == []


def test_cache_keys_separated_by_query(monkeypatch, tmp_path: Path) -> None:
    """Different queries hit different cache files."""
    _isolate_cache(monkeypatch, tmp_path)
    _patch_http(monkeypatch, [{"index": 1, "description": "a", "score": 1.0}])
    sae_labels.search_features_by_label("gemma-2-9b", 20, "alpha")
    _patch_http(monkeypatch, [{"index": 2, "description": "b", "score": 1.0}])
    rows = sae_labels.search_features_by_label("gemma-2-9b", 20, "beta")
    assert rows[0].feature_id == 2
    cache_files = list((tmp_path / "np_cache").glob("*.json"))
    assert len(cache_files) == 2
