"""Tests for the deterministic IOI Stage 0 bridge."""

from __future__ import annotations

import json
from pathlib import Path

from autointerp.pipelines.investigation.ioi_stage0_bridge import bridge_stage0
from autointerp.pipelines.investigation.state import read_state


def test_bridge_stage0_with_precomputed_compact_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ioi_stage0_metric_inputs_compact.json"
    metrics_path.write_text(
        json.dumps(
            {
                "compact_accuracy_inputs": {"n_correct": 498, "n_total": 500},
                "compact_logit_diff_inputs": {"mean_diff": 3.0129375, "n_total": 500},
                "summary": {
                    "accuracy": 0.996,
                    "mean_logit_diff": 3.0129375,
                    "n": 500,
                    "n_correct": 498,
                },
                "prompt_records": [
                    {
                        "template": "ABBA",
                        "prompt": "When Alice and Bob went to the park, Bob gave a book to",
                        "corrupted_prompt": "When Alice and Carol went to the park, Bob gave a book to",
                        "io": " Alice",
                        "s": " Bob",
                        "A": " Alice",
                        "B": " Bob",
                        "C": " Carol",
                        "place": " park",
                        "object": " book",
                    }
                ],
            }
        )
    )

    result = bridge_stage0(
        spec=Path("examples/specs/ioi-gpt2-small-blind-v1_rev1.json"),
        runs_root=tmp_path / "runs",
        metrics_json=metrics_path,
    )

    assert result["status"] == "advanced"
    assert result["accuracy"] == 0.996
    assert result["mean_logit_diff"] == 3.0129375
    assert result["current_stage_idx"] == 1
    assert result["terminal_state"] is None
    assert any(ref.endswith("metric_stage0_accuracy.json") for ref in result["artifact_refs"])
    assert any(ref.endswith("metric_stage0_logit_diff.json") for ref in result["artifact_refs"])
    assert any(ref.endswith("ioi_stage0_dev_pairs.json") for ref in result["artifact_refs"])

    run_root = Path(result["run_root"])
    state = read_state(run_root / "state.json")
    assert state.current_stage_idx == 1
    assert state.criteria_evaluated["behavioral-sanity"].passed is True
    prompt_batch = json.loads((run_root / "prompt_batches/ioi_stage0_dev_pairs.json").read_text())
    assert prompt_batch["cases"][0]["metadata"]["corrupted_prompt"].startswith("When Alice")


def test_bridge_stage0_is_idempotent_after_advancement(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "compact_accuracy_inputs": {"n_correct": 498, "n_total": 500},
                "compact_logit_diff_inputs": {"mean_diff": 3.0129375, "n_total": 500},
                "summary": {
                    "accuracy": 0.996,
                    "mean_logit_diff": 3.0129375,
                    "n": 500,
                    "n_correct": 498,
                },
            }
        )
    )
    kwargs = {
        "spec": Path("examples/specs/ioi-gpt2-small-blind-v1_rev1.json"),
        "runs_root": tmp_path / "runs",
        "metrics_json": metrics_path,
    }

    bridge_stage0(**kwargs)
    result = bridge_stage0(**kwargs)

    assert result["status"] == "already_advanced"
    assert result["current_stage_idx"] == 1
