"""Deterministic session-file writers for the discovery sub-agent.

Mirrors ``auto_circuit_discovery/session.py``: each iteration appends to
``experiments.jsonl`` and lands new ``results/<candidate>/summary.json``
artifacts; this module re-builds ``leaderboard.md`` and ``memory_summary.md``
from those so the next iteration's prompt has a stable, deduplicated view.

Reward field: ``objective_value`` (set by ``evaluate.py`` from the chosen
reward metric — defaults to ``combined_auc_k``).
"""

from __future__ import annotations

import json
from pathlib import Path


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_candidate_results(session_dir: Path) -> list[dict]:
    out: list[dict] = []
    results_dir = session_dir / "results"
    if not results_dir.exists():
        return out
    for path in sorted(results_dir.glob("*/summary.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        out.append(
            {
                "candidate": path.parent.name,
                "path": str(path),
                "objective_metric": payload.get("objective_metric", "combined_auc_k"),
                "objective_value": _safe_float(payload.get("objective_value")),
                "subscores": payload.get("subscores", {}),
                "top_features": payload.get("top_features", []),
            }
        )
    out.sort(key=lambda r: r["objective_value"], reverse=True)
    return out


def _load_experiments(session_dir: Path) -> list[dict]:
    path = session_dir / "experiments.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def write_leaderboard(session_dir: str | Path) -> Path:
    session_dir = Path(session_dir)
    results = _load_candidate_results(session_dir)
    out_path = session_dir / "leaderboard.md"
    lines = [
        "# Leaderboard",
        "",
        "| Rank | Candidate | Objective | Value | Subscores |",
        "|------|-----------|-----------|-------|-----------|",
    ]
    if not results:
        lines.append("| - | none yet | - | - | - |")
    for idx, item in enumerate(results, start=1):
        mark = " **BEST**" if idx == 1 else ""
        sub = ", ".join(f"{k}={v:.4f}" for k, v in item["subscores"].items())
        lines.append(
            f"| {idx} | {item['candidate']} | {item['objective_metric']} | "
            f"{item['objective_value']:.4f} | {sub} |{mark}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_memory_summary(session_dir: str | Path) -> Path:
    session_dir = Path(session_dir)
    results = _load_candidate_results(session_dir)
    experiments = _load_experiments(session_dir)
    out_path = session_dir / "memory_summary.md"
    lines = ["# Research Memory", ""]
    if results:
        lines.append("## Best Candidates")
        for item in results[:5]:
            lines.append(
                f"- `{item['candidate']}`: {item['objective_metric']}={item['objective_value']:.4f}"
            )
        lines.append("")
        best = results[0]
        sub = best.get("subscores") or {}
        if "mean_ablation_auc_k" in sub and "mean_steering_auc_k" in sub:
            lines.append("## Current Signal")
            if sub["mean_ablation_auc_k"] >= sub["mean_steering_auc_k"]:
                lines.append(
                    "- Ablation is currently stronger than steering. Consider "
                    "algorithms that emphasize causal-suppression features."
                )
            else:
                lines.append(
                    "- Steering is currently stronger than ablation. Consider "
                    "algorithms that emphasize amplification features."
                )
            lines.append("")
    if experiments:
        lines.append("## Recent Experiment Notes")
        for record in experiments[-5:]:
            ident = record.get("id", "?")
            hypothesis = (record.get("hypothesis") or "").strip()
            conclusion = (record.get("conclusion") or "").strip()
            parts = [f"`{ident}`"]
            if hypothesis:
                parts.append(f"hypothesis: {hypothesis}")
            if conclusion:
                parts.append(f"conclusion: {conclusion}")
            lines.append(f"- {'; '.join(parts)}")
        lines.append("")
    if not results and not experiments:
        lines.append("No evaluated candidates yet.")
        lines.append("")
    lines.append("## Next Move")
    if results:
        lines.append(
            "- Start from the current best candidate and change one idea at a time."
        )
    else:
        lines.append(
            "- Copy `algorithm_template.py`, run the evaluator, and log the result."
        )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def update_session_files(session_dir: str | Path) -> list[Path]:
    session_dir = Path(session_dir)
    return [write_leaderboard(session_dir), write_memory_summary(session_dir)]


__all__ = [
    "update_session_files",
    "write_leaderboard",
    "write_memory_summary",
]
