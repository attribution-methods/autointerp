"""Tests for evaluator resolution (module + file path), the zero-effect sanity
guard, and that the shipped example evaluators import cleanly (no model load)."""

from __future__ import annotations

import json

import pytest

from autointerp.pipelines.investigation.discovery import harness

_TEMPLATE = "src/autointerp/pipelines/investigation/discovery/algorithm_template.py"


def test_resolve_module_form() -> None:
    fn = harness._resolve_evaluator(
        "autointerp.pipelines.investigation.discovery.evaluators.ioi_heads:evaluate")
    assert callable(fn)


def test_resolve_file_path_form(tmp_path) -> None:
    p = tmp_path / "my_eval.py"
    p.write_text("def evaluate(score_fn, *, top_k, k_grid, seed=0):\n    return {'ok': True}\n")
    fn = harness._resolve_evaluator(f"{p}:evaluate")
    assert callable(fn) and fn(None, top_k=1, k_grid=[1, 2]) == {"ok": True}


def test_resolve_bad_ref() -> None:
    with pytest.raises(SystemExit):
        harness._resolve_evaluator("no_colon_here")
    with pytest.raises(SystemExit):
        harness._resolve_evaluator("/does/not/exist.py:evaluate")


def test_zero_effect_emits_sanity_warning(tmp_path) -> None:
    # A real (non-dry-run) evaluator returning all-zeros must be flagged.
    ev = tmp_path / "zero_eval.py"
    ev.write_text(
        "def evaluate(score_fn, *, top_k, k_grid, seed=0):\n"
        "    return {'mean_ablation_auc_k': 0.0, 'mean_steering_auc_k': 0.0, "
        "'top_features': [], 'k_grid': k_grid}\n"
    )
    out = tmp_path / "out.json"
    rc = harness.main([
        "--algorithm", _TEMPLATE, "--output", str(out),
        "--evaluator", f"{ev}:evaluate", "--k-values", "1,2,3",
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["objective_value"] == 0.0
    assert "_sanity_warning" in payload  # the guard fired


def test_dry_run_has_no_sanity_warning(tmp_path) -> None:
    out = tmp_path / "out.json"
    harness.main(["--algorithm", _TEMPLATE, "--output", str(out), "--dry-run"])
    payload = json.loads(out.read_text())
    assert "_sanity_warning" not in payload  # dry-run is exempt


def test_example_evaluators_import_without_model() -> None:
    # Importing the example modules must be cheap (heavy imports are deferred).
    import importlib
    for mod in ("ioi_heads", "surprise"):
        m = importlib.import_module(
            f"autointerp.pipelines.investigation.discovery.evaluators.{mod}")
        assert callable(m.evaluate)
