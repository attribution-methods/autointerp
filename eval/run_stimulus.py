"""Scaffold-faithfulness eval driver — run one stimulus's 5 conditions.

For a stimulus (e.g. ``s2_ioi``) this runs, sequentially (cost control):

  C0      : python -m autointerp_agent.free_agent      (no scaffold)
  full    : python -m autointerp_agent.investigation --ablation full
  loo-A   : ...                                        --ablation A
  loo-B   : ...                                        --ablation B
  loo-C   : ...                                        --ablation C

All runs: model claude-sonnet-4-5, max_iterations 80, wallclock 1800 s
(decision D3), cwd = repo root, PYTHONPATH=src (env-checkout-mismatch),
each condition in its own --runs-root. A preflight refuses to spend budget
if transformers / the API key are not usable.

Usage:
  PYTHONPATH=src python eval/run_stimulus.py s2_ioi --dry-run
  PYTHONPATH=src python eval/run_stimulus.py s2_ioi --only c0
  PYTHONPATH=src python eval/run_stimulus.py s2_ioi
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
# Routed via OpenRouter (OPENROUTER_API_KEY in repo .env). Same underlying
# model (Anthropic Sonnet 4.5) → no model-identity confound; only the
# billing/routing path differs. context._caching_enabled() recognizes the
# openrouter/anthropic/claude prefix, so prompt caching still applies.
MODEL = "openrouter/anthropic/claude-sonnet-4.5"
MAX_ITERS = "80"
WALLCLOCK = "1800"

# condition -> ("c0" | ablation code for the investigation CLI)
CONDITIONS = {
    "c0": "c0",
    "full": "full",
    "loo-A": "A",
    "loo-B": "B",
    "loo-C": "C",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(SRC)
    existing = env.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


def preflight() -> list[str]:
    """Return a list of blocker strings ([] = good to go)."""
    blockers: list[str] = []
    env = _child_env()
    r = subprocess.run(
        [sys.executable, "-c", "import transformers"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        blockers.append(
            "transformers cannot import (tokenizers version mismatch): "
            + r.stderr.strip().splitlines()[-1]
        )
    r = subprocess.run(
        [sys.executable, "-c",
         "from dotenv import load_dotenv,find_dotenv;import os;"
         "load_dotenv(find_dotenv(usecwd=True),override=False);"
         "import sys;sys.exit(0 if os.environ.get('ANTHROPIC_API_KEY') else 1)"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        blockers.append("ANTHROPIC_API_KEY not resolvable (.env not found / unset)")
    return blockers


def _cmd(stimulus: str, cond: str, out_root: Path) -> list[str]:
    spec = f"eval/specs/{stimulus}.json"
    if cond == "c0":
        return [
            sys.executable, "-m", "autointerp_agent.free_agent",
            "--question-file", f"eval/specs/{stimulus}.question.md",
            "--run-dir", str(out_root / "c0"),
            "--model", MODEL, "--max-iterations", MAX_ITERS,
            "--wallclock", WALLCLOCK, "--quiet",
        ]
    return [
        sys.executable, "-m", "autointerp_agent.investigation",
        "--spec", spec, "--ablation", CONDITIONS[cond],
        "--runs-root", str(out_root / cond),
        "--model", MODEL, "--max-iterations", MAX_ITERS,
        "--wallclock", WALLCLOCK, "--auto-approve", "--quiet",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one eval stimulus (5 conditions)")
    ap.add_argument("stimulus", help="e.g. s2_ioi (matches eval/specs/<stimulus>.json)")
    ap.add_argument("--out-root", default=None,
                    help="default: runs_eval/<stimulus>")
    ap.add_argument("--only", nargs="*", choices=list(CONDITIONS),
                    help="run only these conditions")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="wipe an existing condition dir and re-run it")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args(argv)

    spec_path = REPO / "eval" / "specs" / f"{args.stimulus}.json"
    if not spec_path.exists():
        print(f"no spec at {spec_path}", file=sys.stderr)
        return 2
    out_root = Path(args.out_root) if args.out_root else REPO / "runs_eval" / args.stimulus
    out_root.mkdir(parents=True, exist_ok=True)
    conds = args.only or list(CONDITIONS)

    if not args.dry_run and not args.skip_preflight:
        blockers = preflight()
        if blockers:
            print("PREFLIGHT FAILED — refusing to spend budget:", file=sys.stderr)
            for b in blockers:
                print("  - " + b, file=sys.stderr)
            return 3

    index_path = out_root / "INDEX.json"
    index: dict = json.loads(index_path.read_text()) if index_path.exists() else {}
    env = _child_env()

    for cond in conds:
        cond_dir = out_root / cond
        cmd = _cmd(args.stimulus, cond, out_root)
        if args.dry_run:
            print(f"[{cond}] " + " ".join(cmd))
            continue
        if cond_dir.exists() and any(cond_dir.iterdir()):
            if not args.force:
                print(f"[{cond}] exists, skipping (use --force to re-run)")
                continue
            shutil.rmtree(cond_dir)
        cond_dir.mkdir(parents=True, exist_ok=True)
        log_path = cond_dir / "driver.log"
        print(f"[{cond}] start {_now()}  -> {log_path}")
        t0 = time.monotonic()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, text=True)
        secs = round(time.monotonic() - t0, 1)
        index[cond] = {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "seconds": secs,
            "ended_at": _now(),
            "log": str(log_path.relative_to(REPO)),
        }
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        status = "ok" if proc.returncode == 0 else f"rc={proc.returncode}"
        print(f"[{cond}] done  {secs}s  {status}")

    if not args.dry_run:
        print(f"\nINDEX: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
