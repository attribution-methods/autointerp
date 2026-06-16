"""The push-button causal-capture glue: produce model_forward captures that the
causal metrics + the provenance gate accept. Torch-guarded (CI core install
lacks torch); runs a tiny real gpt2 pass locally."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from autointerp.pipelines.investigation.metrics import (  # noqa: E402
    _ablation_drop,
    _patch_effect_recovery,
)
from autointerp.tools.causal_metrics import (  # noqa: E402
    ablation_drop_capture,
    best_patch_site,
    best_patch_sites,
    circuit_recovery_capture,
    patch_recovery_capture,
)
from autointerp.tools.model import load_model  # noqa: E402
from autointerp.tools.provenance import capture_for_relpath, verify_capture  # noqa: E402


def _metric_fns(handle):
    tok = handle.tokenizer
    a = tok(" Mary").input_ids[0]
    b = tok(" John").input_ids[0]
    return lambda logits: logits[:, a] - logits[:, b]


def test_patch_and_ablation_captures_pass_the_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOINTERP_RUN_DIR", str(tmp_path))
    (tmp_path / "captures").mkdir(exist_ok=True)
    h = load_model("gpt2", device="cpu")
    metric = _metric_fns(h)
    clean = ["When John and Mary went to the store, John gave a drink to",
             "When Alice and Bob met, Alice handed the book to"]
    corrupt = ["When John and John went to the store, John gave a drink to",
               "When Bob and Bob met, Alice handed the book to"]

    site = best_patch_site(h, clean, corrupt, metric, layers=list(range(6, 11)))
    assert isinstance(site, tuple) and len(site) == 2

    rel = patch_recovery_capture(h, clean, corrupt, site, metric, model_id="gpt2")
    cap = capture_for_relpath(str(tmp_path), rel)
    assert cap is not None and cap["source"] == "model_forward"
    assert verify_capture(str(tmp_path), cap)  # unmodified, ledgered
    data = json.loads((tmp_path / cap["data_relpath"]).read_text())
    assert {"clean_metric", "corrupt_metric", "patched_metric"} <= set(data)
    # consumable by the metric (no exception, finite float)
    assert isinstance(_patch_effect_recovery(data), float)

    rel2 = ablation_drop_capture(h, clean, [site], metric, model_id="gpt2")
    cap2 = capture_for_relpath(str(tmp_path), rel2)
    assert cap2 is not None and cap2["source"] == "model_forward"
    data2 = json.loads((tmp_path / cap2["data_relpath"]).read_text())
    assert {"baseline_metric", "ablated_metric"} <= set(data2)
    assert isinstance(_ablation_drop(data2), float)


def test_circuit_recovery_capture_multi_site(tmp_path, monkeypatch):
    """A distributed circuit (IOI name movers) must be patched as a SET — one
    head recovers ≈0. circuit_recovery_capture discovers the set push-button and
    records a gate-passing capture; multi-site recovery beats the single head."""
    monkeypatch.setenv("AUTOINTERP_RUN_DIR", str(tmp_path))
    (tmp_path / "captures").mkdir(exist_ok=True)
    h = load_model("gpt2", device="cpu")
    metric = _metric_fns(h)
    # repeated-subject clean vs ABC corrupt — the standard IOI contrast.
    clean = ["When Mary and John went to the store, John gave a drink to",
             "When Mary and John went to the store, John handed a book to"]
    corrupt = ["When Mary and John went to the store, Carl gave a drink to",
               "When Mary and John went to the store, Carl handed a book to"]
    layers = list(range(8, 12))

    sites = best_patch_sites(h, clean, corrupt, metric, layers=layers, top_k=3)
    assert isinstance(sites, list) and 1 <= len(sites) <= 3
    assert all(isinstance(s, tuple) and len(s) == 2 for s in sites)

    rel = circuit_recovery_capture(
        h, clean, corrupt, metric, layers=layers, top_k=3, model_id="gpt2"
    )
    cap = capture_for_relpath(str(tmp_path), rel)
    assert cap is not None and cap["source"] == "model_forward"
    assert verify_capture(str(tmp_path), cap)
    data = json.loads((tmp_path / cap["data_relpath"]).read_text())
    assert len(data["senders"]) == len(sites)  # whole set recorded
    assert isinstance(_patch_effect_recovery(data), float)
    # patching the set recovers strictly more than the single best head alone.
    one = patch_recovery_capture(h, clean, corrupt, sites[0], metric, model_id="gpt2")
    one_cap = capture_for_relpath(str(tmp_path), one)
    one_data = json.loads((tmp_path / one_cap["data_relpath"]).read_text())
    assert data["recovery"] > one_data["recovery"]
