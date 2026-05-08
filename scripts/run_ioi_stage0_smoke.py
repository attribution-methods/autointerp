#!/usr/bin/env python3
"""Compact IOI Stage 0 behavioral check for GPT-2-small."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autointerp.pipelines.ioi_stage0 import run_ioi_stage0_forward, write_metric_inputs


def _dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError(f"unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="scratch/ioi_stage0_metric_inputs_compact.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model/tokenizer downloads instead of requiring cached files.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also write raw predictions/logits. Avoid this in LLM tool-call flows.",
    )
    args = parser.parse_args()

    out = run_ioi_stage0_forward(
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        local_files_only=not args.allow_download,
        include_raw=args.include_raw,
    )
    summary = out["summary"]
    print(f"names={summary['n_names']} prompts={summary['n']}")
    print(f"accuracy={summary['accuracy']:.4f}")
    print(f"mean_logit_diff={summary['mean_logit_diff']:.4f}")
    print("first_prompt=", summary["first_prompt"])
    print("first_diff=", round(float(summary["first_diff"]), 4))
    path = write_metric_inputs(args.output, out)
    print(f"wrote={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
