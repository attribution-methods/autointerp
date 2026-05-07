"""End-to-end dry-run tests for the discovery sub-agent loop.

These never invoke the SDK or any LLM. They exercise the full
init→prompt→harness round-trip via the deterministic stub evaluator.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from autointerp.pipelines.investigation.discovery import run_discovery_subagent


@pytest.fixture(autouse=True)
def _ensure_pythonpath(monkeypatch):
    """Subprocess harness needs `src` on PYTHONPATH for autointerp imports."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    pp = os.environ.get("PYTHONPATH", "")
    if src not in pp.split(os.pathsep):
        pp = os.pathsep.join([src, pp]) if pp else src
    monkeypatch.setenv("PYTHONPATH", pp)


def test_discovery_dry_run_round_trip(tmp_path: Path):
    res = run_discovery_subagent(
        session_dir=tmp_path / "session_a",
        task="demo discovery task",
        reward_metric="combined_auc_k",
        reward_description="0.5 * (ablation + steering); higher is better.",
        max_iterations=3,
        dry_run=True,
    )
    assert res.terminated_by == "done", res
    assert res.iterations_run == 1
    assert res.best_candidate == "algorithm_v1"
    assert res.best_summary is not None
    assert 0.0 <= res.best_summary["objective_value"] <= 1.0


def test_discovery_artifacts_are_written(tmp_path: Path):
    session = tmp_path / "session_b"
    res = run_discovery_subagent(
        session_dir=session,
        task="demo",
        reward_metric="combined_auc_k",
        reward_description="combined",
        max_iterations=2,
        dry_run=True,
    )
    assert res.session_dir == session.resolve()
    # Required session files exist with non-empty content.
    for name in (
        "scratchpad.md",
        "experiments.jsonl",
        "leaderboard.md",
        "memory_summary.md",
        "algorithm_template.py",
        "algorithm_v1.py",
    ):
        assert (session / name).exists(), name
    # Harness ran and wrote a result for the candidate.
    summary = session / "results" / "algorithm_v1" / "summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text())
    assert payload["objective_metric"] == "combined_auc_k"
    assert "subscores" in payload
    # Experiments log captured at least the dry-run baseline entry.
    log = (session / "experiments.jsonl").read_text().splitlines()
    assert log
    parsed = json.loads(log[0])
    assert parsed["id"] == "algorithm_v1"
    assert "result" in parsed


def test_discovery_terminates_on_done_marker(tmp_path: Path):
    """Even with budget for many iterations, [DONE] in scratchpad halts."""
    res = run_discovery_subagent(
        session_dir=tmp_path / "session_c",
        task="demo",
        reward_metric="auroc",
        reward_description="ROC AUC for a binary readout",
        max_iterations=10,
        dry_run=True,
    )
    # Dry-run iteration always writes [DONE] at the end.
    assert res.terminated_by == "done"
    assert res.iterations_run == 1
