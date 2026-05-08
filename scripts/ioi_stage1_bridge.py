#!/usr/bin/env python3
"""Bridge IOI Stage 1 with head-level patch recovery metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autointerp.pipelines.investigation.ioi_stage1_bridge import bridge_stage1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="examples/specs/ioi-gpt2-small-blind-v1_rev1.json",
        help="Approved InvestigationSpec JSON.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs_gpt54mini_gpu_bridge",
        help="Run root containing the IOI run dir.",
    )
    parser.add_argument("--results-json", help="Precomputed Stage 1 result JSON.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--skip-criterion",
        action="store_true",
        help="Commit Stage 1 metrics but do not evaluate localization-recovery.",
    )
    args = parser.parse_args()

    result = bridge_stage1(
        spec=args.spec,
        runs_root=args.runs_root,
        results_json=args.results_json,
        top_k=args.top_k,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        allow_download=args.allow_download,
        evaluate_localization=not args.skip_criterion,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
