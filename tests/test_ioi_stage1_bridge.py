"""Tests for the deterministic IOI Stage 1 bridge."""

from __future__ import annotations

import json
from pathlib import Path

from autointerp.pipelines.investigation.ioi_stage0_bridge import bridge_stage0
from autointerp.pipelines.investigation.ioi_stage1_bridge import bridge_stage1
from autointerp.pipelines.investigation.state import read_state


def _stage0_metrics(path: Path) -> None:
    path.write_text(
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


def test_bridge_stage1_with_precomputed_results(tmp_path: Path) -> None:
    spec = Path("examples/specs/ioi-gpt2-small-blind-v1_rev1.json")
    runs_root = tmp_path / "runs"
    stage0_path = tmp_path / "stage0.json"
    _stage0_metrics(stage0_path)
    bridge_stage0(spec=spec, runs_root=runs_root, metrics_json=stage0_path)

    results_path = tmp_path / "stage1.json"
    results_path.write_text(
        json.dumps(
            {
                "clean_metric": 3.0,
                "corrupt_metric": 0.0,
                "top_k_patched_metric": 2.7,
                "top_k_recovery": 0.9,
                "top_k_kl_per_sample": [0.1, 0.2, 0.3],
                "n_samples": 3,
                "top_k": 2,
                "top_k_heads": [
                    {"layer": 9, "head": 6, "patched_metric": 2.0, "recovery": 0.66},
                    {"layer": 10, "head": 0, "patched_metric": 1.8, "recovery": 0.60},
                ],
            }
        )
    )

    result = bridge_stage1(spec=spec, runs_root=runs_root, results_json=results_path)

    assert result["status"] == "advanced"
    assert result["top_k_recovery"] == 0.9
    assert result["current_stage_idx"] == 2
    assert result["terminal_state"] is None
    assert any(ref.endswith("metric_stage1_kl_to_clean.json") for ref in result["artifact_refs"])
    assert any(ref.endswith("metric_stage1_patch_effect_recovery.json") for ref in result["artifact_refs"])
    assert any(ref.endswith("candidate_stage1_L9H6.json") for ref in result["artifact_refs"])

    state = read_state(Path(result["run_root"]) / "state.json")
    assert state.current_stage_idx == 2
    assert state.criteria_evaluated["localization-recovery"].passed is True
