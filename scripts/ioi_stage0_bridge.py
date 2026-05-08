#!/usr/bin/env python3
"""Bridge IOI Stage 0 with compact canonical metrics and advance to Stage 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autointerp.pipelines.investigation.ioi_stage0_bridge import bridge_stage0


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
        help="Run root under which the IOI run dir is created/resumed.",
    )
    parser.add_argument("--metrics-json", help="Precomputed compact metric input JSON.")
    parser.add_argument("--resume", action="store_true", help="Require an existing run.")
    parser.add_argument("--no-resume", action="store_true", help="Require a fresh run.")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--min-logit-diff", type=float, default=2.0)
    args = parser.parse_args()

    resume: bool | None
    if args.resume:
        resume = True
    elif args.no_resume:
        resume = False
    else:
        resume = None

    result = bridge_stage0(
        spec=args.spec,
        runs_root=args.runs_root,
        resume=resume,
        metrics_json=args.metrics_json,
        min_accuracy=args.min_accuracy,
        min_logit_diff=args.min_logit_diff,
        include_raw=args.include_raw,
        allow_download=args.allow_download,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
