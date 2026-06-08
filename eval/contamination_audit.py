"""Contamination audit — mandatory post-run gate for the scaffold eval.

Scans every agent-controlled surface of a finished run dir for any reference
to answer-bearing locations that must never have been reachable: the ORIGINAL
repo tree, the out-of-tree scoring keys, prior solved runs, priors/, logs,
RESULTS/SCORE artifacts, sibling RQ1 runs.

This detects *environment leakage* (the agent reading something it shouldn't),
NOT whether the agent recited a circuit from its own training — the latter is
the RQ3 measurement, decided by scoring, not here.

Exit code 0 = clean, 1 = contaminated (caller must void the condition).

Usage:
  python eval/contamination_audit.py <run_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# The ORIGINAL repo and the out-of-tree keys. The clean room lives at
# .../attribution-methods/eval_cleanroom/repo and must NOT match these.
ORIG_REPO = "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/"
KEYS_DIR = "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp_eval_keys"

# Substrings/patterns that, appearing anywhere the agent wrote or read, mean
# it touched a forbidden location. Path/filename based on purpose — precise,
# deterministic, no false-positives on the agent's own circuit prose.
FORBIDDEN = [
    re.escape(ORIG_REPO),                 # original repo abs path (clean room differs)
    re.escape(KEYS_DIR),
    r"autointerp_eval_keys",
    r"/priors\b", r"priors/ioi", r"\bioi\.yaml\b",
    # prior solved-run / proven-spec ids ONLY — must NOT match the sanctioned
    # stimulus id `eval-ioi-blind-v1` (legitimately in spec.json/progress.md).
    r"ioi-gpt2-small-blind", r"logs/ioi-blind",
    r"/runs/ioi", r"runs/_ioi",
    r"\bruns_eval\b",
    r"RESULTS\.md", r"SCORE\.md", r"score\.json",
    r"c0_bf16", r"/c0/", r"\bc0\b/scratch",
    r"/logs/ioi", r"/examples/specs/ioi", r"/outputs/specs/ioi",
]
_RX = re.compile("|".join(f"(?:{p})" for p in FORBIDDEN))

# Surfaces the agent controls or that capture its I/O verbatim.
SCAN_GLOBS = [
    "tool_invocations.jsonl",
    "tool_invocations/*",
    "scratch/bash_*.log",
    "scratch/*.md",
    "scripts/*",
    "assistant_turns.jsonl",
    "INVESTIGATION_LOG.md",
    "progress.md",
    "report.json",
    "findings/*",
]


def audit(run_dir: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for pattern in SCAN_GLOBS:
        for f in sorted(run_dir.glob(pattern)):
            if not f.is_file():
                continue
            try:
                text = f.read_text(errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if _RX.search(line):
                    hits.append((str(f.relative_to(run_dir)), i, line.strip()[:240]))
    return hits


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: contamination_audit.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    if not run_dir.is_dir():
        print(f"not a dir: {run_dir}", file=sys.stderr)
        return 2
    hits = audit(run_dir)
    if not hits:
        print(f"CLEAN  {run_dir}  (no forbidden-location references)")
        return 0
    print(f"CONTAMINATED  {run_dir}  ({len(hits)} hit(s)) — condition VOID:")
    for rel, ln, snippet in hits[:60]:
        print(f"  {rel}:{ln}: {snippet}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
