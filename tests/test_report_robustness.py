"""Report assembly must never crash on stray/off-schema files in the scanned
artifact directories — the agent can write arbitrary JSON there (e.g. a
hand-rolled activation-cache sidecar). Regression test for the uncaught
ValidationError in `_load_all`.
"""

from __future__ import annotations

import json

import pytest

from autointerp import schemas as S
from autointerp.pipelines.investigation.report import _load_all


def _valid_cache_ref() -> S.ActivationCacheRef:
    return S.ActivationCacheRef(
        cache_id="c1",
        model=S.ModelRef(model_id="gpt2"),
        source_path="activations/c1.npz",
        prompt_batch_id="pb1",
        layers=[8, 9, 10],
    )


def test_load_all_skips_off_schema_and_keeps_valid(tmp_path):
    # one valid artifact...
    (tmp_path / "good.json").write_text(_valid_cache_ref().model_dump_json())
    # ...one off-schema sidecar the agent dumped (the real-world trigger)...
    (tmp_path / "dev_cache_L16_metadata.json").write_text(json.dumps({
        "split": "dev", "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "layer": 16,
        "n_samples": 50, "component": "resid", "token_index": -1,
        "hidden_dim": 1536, "sample_range": "50-100",
    }))
    # ...and one non-JSON file.
    (tmp_path / "broken.json").write_text("not json {{{")

    with pytest.warns(UserWarning, match="off-schema"):
        out = _load_all(tmp_path, "*.json", S.ActivationCacheRef)

    # No crash; only the valid artifact is returned.
    assert len(out) == 1
    assert out[0].cache_id == "c1"


def test_load_all_empty_dir(tmp_path):
    assert _load_all(tmp_path / "nope", "*.json", S.ActivationCacheRef) == []


def test_load_all_all_off_schema_returns_empty(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"foo": "bar"}))
    with pytest.warns(UserWarning, match="off-schema"):
        out = _load_all(tmp_path, "*.json", S.ActivationCacheRef)
    assert out == []
