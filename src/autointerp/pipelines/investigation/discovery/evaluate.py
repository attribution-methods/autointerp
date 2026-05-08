"""Discovery evaluator — invoked by the sub-agent each iteration.

Stable bridge between the agentic loop and the harness. Mirrors
``circuitbreaker.auto_circuit_discovery.evaluate``:

1. Resolve the candidate ``algorithm_v{N}.py`` path.
2. Spawn ``harness.py`` as a subprocess so the candidate's import side
   effects don't leak into the loop's process.
3. Read the harness JSON, write a small ``summary.json`` with just the
   objective value (the loop summary reads this; full body stays in
   ``harness_result.json``).
4. Trigger ``session.update_session_files`` so ``leaderboard.md`` /
   ``memory_summary.md`` are refreshed for the next iteration prompt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# We deliberately don't import heavy autointerp modules at top level: this
# script runs as a subprocess and should start fast.

_HARNESS = Path(__file__).resolve().parent / "harness.py"


def _resolve(session_dir: Path, raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (session_dir / p).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one discovery candidate")
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--algorithm", required=True,
                        help="Candidate path, relative to session-dir or absolute")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--objective", default="combined")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--k-values", default="1,5,10,20,50")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--evaluator", default=None,
                        help="Python ref `module:attr` for a real evaluator.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use the stub evaluator; no model load.")
    # Substrate / benchmark wiring (real-path only).
    parser.add_argument("--substrate", default=None,
                        choices=[None, "components", "features"])
    parser.add_argument("--component-kind", default="attn_head")
    parser.add_argument("--decomposition", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="dev")
    args = parser.parse_args(argv)

    session_dir = args.session_dir.expanduser().resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = _resolve(session_dir, args.algorithm)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate not found: {candidate_path}")

    results_dir = session_dir / "results" / args.candidate_name
    results_dir.mkdir(parents=True, exist_ok=True)
    harness_out = results_dir / "harness_result.json"
    summary_out = results_dir / "summary.json"

    cmd: list[str] = [
        sys.executable, str(_HARNESS),
        "--algorithm", str(candidate_path),
        "--output", str(harness_out),
        "--top-k", str(args.top_k),
        "--k-values", args.k_values,
        "--objective", args.objective,
    ]
    if args.layers:
        cmd += ["--layers", args.layers]
    if args.evaluator is not None:
        cmd += ["--evaluator", args.evaluator]
    if args.dry_run:
        cmd.append("--dry-run")
    else:
        # Real path requires substrate + benchmark + model id.
        if args.substrate is not None:
            cmd += ["--substrate", args.substrate]
        if args.substrate == "components":
            cmd += ["--component-kind", args.component_kind]
        if args.substrate == "features" and args.decomposition:
            cmd += ["--decomposition", args.decomposition]
        if args.benchmark:
            cmd += ["--benchmark", args.benchmark]
        if args.model_id:
            cmd += ["--model-id", args.model_id]
        if args.device:
            cmd += ["--device", args.device]
        cmd += [
            "--n-pairs", str(args.n_pairs),
            "--seed", str(args.seed),
            "--split", args.split,
        ]

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Harness failed (exit={proc.returncode}) for {args.candidate_name}")

    payload = json.loads(harness_out.read_text())
    summary = {
        "candidate_name": args.candidate_name,
        "candidate_path": str(candidate_path),
        "result_path": str(harness_out),
        "objective": args.objective,
        "objective_metric": payload.get("objective_metric", "combined_auc_k"),
        "objective_value": float(payload.get("objective_value", 0.0)),
        "subscores": {
            "mean_ablation_auc_k": float(payload.get("mean_ablation_auc_k", 0.0)),
            "mean_steering_auc_k": float(payload.get("mean_steering_auc_k", 0.0)),
            "combined_auc_k": float(payload.get("combined_auc_k", 0.0)),
        },
        "top_features": payload.get("top_features", []),
    }
    summary_out.write_text(json.dumps(summary, indent=2))

    # Trigger session-file refresh. Imported lazily so this script runs
    # standalone without sys.path tweaks on the user side.
    try:
        from autointerp.pipelines.investigation.discovery.session import (
            update_session_files,
        )
        update_session_files(session_dir)
    except Exception as exc:  # pragma: no cover — defensive only
        print(f"[evaluate] warning: session refresh failed: {exc}", file=sys.stderr)

    print(f"Candidate: {args.candidate_name}")
    print(f"Result:    {harness_out}")
    print(f"Summary:   {summary_out}")
    print(f"Objective ({args.objective}): {summary['objective_value']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
