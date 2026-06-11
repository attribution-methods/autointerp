"""Hardened, isolated runner for the scaffolded IOI conditions.

Containment model (layout isolation, per design decision):
  * The agent's entire world is the sanitized CLEAN ROOM (allowlist clone:
    only src/skills/configs + the sanitized blinded spec). It has no
    priors/runs/runs_eval/logs/examples/outputs and no scoring keys.
  * For the duration of each agent subprocess the ORIGINAL repo's
    answer-bearing dirs + the out-of-tree keys are chmod-locked (defense in
    depth, instant + reversible) and restored in a finally.
  * After each condition a MANDATORY contamination audit runs; any hit voids
    the condition (recorded; not silently scored).

The agent process: cwd = clean room, PYTHONPATH = cleanroom/src, --spec /
--runs-root inside the clean room. It is never handed the original path.

Usage:
  python eval/run_isolated.py --dry-run
  python eval/run_isolated.py                 # full loo-A loo-B loo-C
  python eval/run_isolated.py --only full
  python eval/run_isolated.py --unlock        # emergency: restore perms only
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

ORIG = Path("/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp")
CR = Path("/lambda/nfs/test-filesystem/fadi/attribution-methods/eval_cleanroom/repo")
KEYS = Path("/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp_eval_keys")
SPEC = CR / "eval/specs/s2_ioi.json"
RUNS_BASE = CR / "active_runs"
INDEX = CR.parent / "INDEX.json"
PERM_SAVE = Path("/tmp/eval_isolated_perms.json")
AUDIT = ORIG / "eval/contamination_audit.py"

MODEL = "openrouter/anthropic/claude-sonnet-4.5"
MAX_ITERS, WALLCLOCK = "80", "1800"
CONDITIONS = {"full": "full", "loo-A": "A", "loo-B": "B", "loo-C": "C"}

# ORIGINAL-repo dirs that hold answers / prior solutions + the keys dir.
LOCK_TARGETS = [
    ORIG / "priors", ORIG / "runs", ORIG / "runs_eval",
    ORIG / "logs", ORIG / "examples", ORIG / "outputs", KEYS,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def lock() -> None:
    saved = {}
    for p in LOCK_TARGETS:
        if p.exists():
            saved[str(p)] = p.stat().st_mode & 0o777
            os.chmod(p, 0o000)
    PERM_SAVE.write_text(json.dumps(saved))
    print(f"  locked {len(saved)} answer dir(s)")


def unlock(force: bool = False) -> None:
    if PERM_SAVE.exists():
        saved = json.loads(PERM_SAVE.read_text())
        for path, mode in saved.items():
            try:
                os.chmod(path, mode)
            except OSError:
                pass
        PERM_SAVE.unlink(missing_ok=True)
        print(f"  restored perms on {len(saved)} dir(s)")
        return
    # No save-file => already restored. Idempotent no-op, UNLESS an explicit
    # emergency --unlock: then force a sane owner-rwx default.
    if force:
        for p in LOCK_TARGETS:
            if p.exists():
                try:
                    os.chmod(p, 0o775)
                except OSError:
                    pass
        print("  emergency unlock -> 0775 (no saved perms found)")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CR / "src")  # clean-room library wins over any editable install
    # Inject API creds from the ORIGINAL .env into the subprocess ENVIRONMENT
    # (not into the clean-room filesystem — the agent never sees a .env).
    dotenv = ORIG / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
                env[k] = v.strip().strip('"').strip("'")
    return env


def run_condition(cond: str, index: dict) -> None:
    code = CONDITIONS[cond]
    # No sibling/solved run may sit in the clean room while another condition
    # runs (a finished run names the circuit → cross-condition answer leak).
    # Start every condition from a guaranteed-empty active_runs.
    if RUNS_BASE.exists():
        for stale in RUNS_BASE.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink()
    runs_root = RUNS_BASE / cond
    runs_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "autointerp_agent.investigation",
        "--spec", str(SPEC), "--ablation", code,
        "--runs-root", str(runs_root),
        "--model", MODEL, "--max-iterations", MAX_ITERS,
        "--wallclock", WALLCLOCK, "--auto-approve", "--quiet",
    ]
    log = runs_root / "driver.log"
    print(f"[{cond}] start {_now()}")
    t0 = time.monotonic()
    lock()
    try:
        with log.open("w") as fh:
            proc = subprocess.run(cmd, cwd=CR, env=child_env(),
                                  stdout=fh, stderr=subprocess.STDOUT, text=True)
        rc = proc.returncode
    finally:
        unlock()  # ALWAYS restore, even on crash/kill
    secs = round(time.monotonic() - t0, 1)

    # mandatory contamination gate
    run_dirs = [d for d in runs_root.iterdir() if d.is_dir()]
    audit_status, audit_out = "no-run-dir", ""
    if run_dirs:
        rd = max(run_dirs, key=lambda d: d.stat().st_mtime)
        a = subprocess.run([sys.executable, str(AUDIT), str(rd)],
                            capture_output=True, text=True)
        audit_status = "clean" if a.returncode == 0 else "CONTAMINATED"
        audit_out = (a.stdout + a.stderr).strip()
    index[cond] = {
        "returncode": rc, "seconds": secs, "ended_at": _now(),
        "audit": audit_status, "audit_detail": audit_out[:4000],
        "runs_root": str(runs_root),
    }
    INDEX.write_text(json.dumps(index, indent=2) + "\n")
    print(f"[{cond}] done {secs}s rc={rc}  audit={audit_status}")
    if audit_status == "CONTAMINATED":
        print(f"[{cond}] !!! VOID — {audit_out.splitlines()[0] if audit_out else ''}")

    # Relocate the finished run OUT of the clean room so a later condition
    # cannot read it (a completed run names the circuit). Destination is under
    # ORIG/runs_eval — chmod-locked during any subsequent agent run and never
    # inside the agent's clean-room world.
    archive = ORIG / "runs_eval" / "s2_ioi" / f"{cond}_run"
    if archive.exists():
        shutil.rmtree(archive, ignore_errors=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(runs_root), str(archive))
    index[cond]["archived_to"] = str(archive)
    INDEX.write_text(json.dumps(index, indent=2) + "\n")
    print(f"[{cond}] archived -> {archive}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CONDITIONS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unlock", action="store_true",
                    help="emergency: restore answer-dir perms and exit")
    args = ap.parse_args(argv)

    if args.unlock:
        unlock(force=True)
        return 0

    for need in (CR, SPEC, AUDIT):
        if not need.exists():
            print(f"missing prerequisite: {need}", file=sys.stderr)
            return 2

    conds = args.only or list(CONDITIONS)
    if args.dry_run:
        for c in conds:
            print(f"[{c}] python -m autointerp_agent.investigation --spec {SPEC} "
                  f"--ablation {CONDITIONS[c]} --runs-root {RUNS_BASE / c} "
                  f"(cwd={CR}, PYTHONPATH={CR/'src'}, answer dirs locked, audit after)")
        return 0

    index: dict = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    try:
        for c in conds:
            run_condition(c, index)
    finally:
        unlock()  # paranoia: never leave the repo locked
    print(f"\nINDEX: {INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
