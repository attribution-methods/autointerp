"""Tests for the discovery session-file writers and the [DONE] sentinel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autointerp.pipelines.investigation.discovery.loop import (
    STATUS_DONE,
    STATUS_NEXT,
    check_done,
    init_session,
)
from autointerp.pipelines.investigation.discovery.session import (
    update_session_files,
    write_leaderboard,
    write_memory_summary,
)


# ---- check_done ------------------------------------------------------------


def test_check_done_initial_next_only():
    assert not check_done(f"{STATUS_NEXT} get going")


def test_check_done_done_after_next():
    text = f"{STATUS_NEXT} step 1\n{STATUS_DONE} all good"
    assert check_done(text)


def test_check_done_next_after_done():
    """Reviving a session: the LAST marker is what matters."""
    text = f"{STATUS_DONE} earlier\n{STATUS_NEXT} actually keep going"
    assert not check_done(text)


def test_check_done_no_markers_at_all():
    assert not check_done("just notes, no markers")


# ---- session writers -------------------------------------------------------


def _seed_summary(session_dir: Path, name: str, *, objective: float,
                  ablation: float, steering: float) -> None:
    rdir = session_dir / "results" / name
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_name": name,
                "objective": "combined",
                "objective_metric": "combined_auc_k",
                "objective_value": objective,
                "subscores": {
                    "mean_ablation_auc_k": ablation,
                    "mean_steering_auc_k": steering,
                    "combined_auc_k": objective,
                },
                "top_features": [
                    {"layer": 1, "feature_id": 42, "score": 0.7},
                ],
            }
        )
    )


def test_leaderboard_orders_by_objective(tmp_path: Path):
    _seed_summary(tmp_path, "algorithm_v1", objective=0.4, ablation=0.5, steering=0.3)
    _seed_summary(tmp_path, "algorithm_v2", objective=0.7, ablation=0.8, steering=0.6)
    write_leaderboard(tmp_path)
    text = (tmp_path / "leaderboard.md").read_text()
    # v2 should land in rank 1 (BEST), v1 below.
    assert "algorithm_v2" in text
    v2_pos = text.index("algorithm_v2")
    v1_pos = text.index("algorithm_v1")
    assert v2_pos < v1_pos
    assert "**BEST**" in text


def test_memory_summary_recommends_ablation_when_higher(tmp_path: Path):
    _seed_summary(tmp_path, "algorithm_v1", objective=0.7, ablation=0.9, steering=0.5)
    write_memory_summary(tmp_path)
    text = (tmp_path / "memory_summary.md").read_text()
    assert "Ablation is currently stronger" in text


def test_memory_summary_recommends_steering_when_higher(tmp_path: Path):
    _seed_summary(tmp_path, "algorithm_v1", objective=0.7, ablation=0.5, steering=0.9)
    write_memory_summary(tmp_path)
    text = (tmp_path / "memory_summary.md").read_text()
    assert "Steering is currently stronger" in text


def test_update_session_files_writes_both(tmp_path: Path):
    _seed_summary(tmp_path, "v1", objective=0.5, ablation=0.5, steering=0.5)
    paths = update_session_files(tmp_path)
    names = {p.name for p in paths}
    assert {"leaderboard.md", "memory_summary.md"} <= names


# ---- init_session ----------------------------------------------------------


def test_init_session_seeds_files(tmp_path: Path):
    init_session(tmp_path, task="demo task", reward_metric="combined_auc_k")
    assert (tmp_path / "scratchpad.md").exists()
    assert (tmp_path / "experiments.jsonl").exists()
    assert (tmp_path / "algorithm_template.py").exists()
    assert (tmp_path / "results").is_dir()
    sp = (tmp_path / "scratchpad.md").read_text()
    assert "demo task" in sp
    assert "combined_auc_k" in sp
    assert STATUS_NEXT in sp
