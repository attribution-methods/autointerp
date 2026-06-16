"""Capture ledger (input-provenance v2): record / verify / tamper-detect, the
env-resolved agent helper, and the bash run-dir env injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from autointerp.tools import provenance as P


def test_record_and_verify(tmp_path: Path) -> None:
    rel = P.record_to(
        tmp_path, "logits",
        {"target_logits": [2.3, 1.0], "foil_logits": [1.2, 0.5]},
        source="model_forward", model_id="gpt2", prompt_batch="b1",
    )
    assert rel.startswith("captures/") and (tmp_path / rel).exists()
    cap = P.capture_for_relpath(tmp_path, rel)
    assert cap is not None
    assert cap["source"] == "model_forward"
    assert cap["model_id"] == "gpt2"
    assert cap["n"] == 2  # inferred sample count
    assert P.verify_capture(tmp_path, cap)


def test_tamper_is_detected(tmp_path: Path) -> None:
    rel = P.record_to(tmp_path, "x", {"a": [1, 2, 3]}, source="model_forward")
    cap = P.capture_for_relpath(tmp_path, rel)
    (tmp_path / rel).write_text('{"a":[9,9,9]}')  # swap the data after recording
    assert not P.verify_capture(tmp_path, cap)


def test_record_capture_resolves_run_dir_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOINTERP_RUN_DIR", str(tmp_path))
    rel = P.record_capture("x", {"a": [1, 2, 3]})
    cap = P.capture_for_relpath(tmp_path, rel)
    assert cap is not None and cap["n"] == 3 and cap["source"] == "model_forward"


def test_record_capture_requires_run_dir(monkeypatch) -> None:
    monkeypatch.delenv("AUTOINTERP_RUN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="AUTOINTERP_RUN_DIR"):
        P.record_capture("x", {"a": [1]})


def test_invalid_source_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source must be"):
        P.record_to(tmp_path, "x", {"a": [1]}, source="bogus")


@pytest.mark.parametrize(
    "value",
    [
        {"prompts": []},                          # empty-prompts stub
        {"placeholder": True, "notes": "stub"},   # flag+note stub
        {},                                       # empty dict
        [],                                       # empty list
        {"predictions": [], "labels": []},        # label-only stub
    ],
)
def test_placeholder_model_forward_capture_rejected(tmp_path: Path, value) -> None:
    """A model_forward capture must carry real measurement data — the lazy
    fabrication path (a stub written just to satisfy the gate) is impossible."""
    with pytest.raises(ValueError, match="no measurement data"):
        P.record_to(tmp_path, "stub", value, source="model_forward")


def test_placeholder_allowed_as_manual(tmp_path: Path) -> None:
    """The same stub is allowed under source='manual' — it just can't ground a
    measurement criterion."""
    rel = P.record_to(tmp_path, "note", {"placeholder": True}, source="manual")
    assert (tmp_path / rel).exists()


def test_real_model_forward_captures_allowed(tmp_path: Path) -> None:
    """Scalars, non-empty arrays, and generated text all pass the guard."""
    for value in (
        0.43,
        {"clean_metric": -0.6, "corrupt_metric": -1.4, "patched_metric": -0.9},
        {"prompts": ["a"], "predictions": [1], "labels": [1]},
        {"generations": ["the cat sat on the mat"]},
    ):
        rel = P.record_to(tmp_path, "real", value, source="model_forward")
        assert (tmp_path / rel).exists()


def test_capture_for_relpath_none_for_plain_file(tmp_path: Path) -> None:
    assert P.capture_for_relpath(tmp_path, "scratch/x.json") is None


def test_bash_injects_run_dir_env(tmp_path: Path) -> None:
    """The agent's bash subprocess must see AUTOINTERP_RUN_DIR so its scripts can
    call record_capture."""
    from autointerp_agent.tools import _run_bash_watched

    run_dir = tmp_path / "runs" / "r1"
    out, ok = _run_bash_watched(
        'echo "RD=$AUTOINTERP_RUN_DIR"', str(tmp_path), 30, 10, None, run_dir
    )
    assert ok
    assert f"RD={run_dir.resolve()}" in out
