#!/usr/bin/env python
"""Install the PyTorch build that matches THIS machine's GPU.

Why this exists: PyPI's default ``torch`` wheel only ships CUDA kernels up to
``sm_90``. On Blackwell GPUs — B200 (``sm_100``) and RTX 50-series
(``sm_120``) — *every* CUDA op then fails with "no kernel image is available
for execution on the device". The CUDA build is chosen by the install *index*
(``download.pytorch.org/whl/cu128`` vs default PyPI), which a plain
``torch>=...`` dependency in ``pyproject.toml`` cannot express. So the
``mechinterp`` extra gets you *a* torch; this script gets you the *right* one.

Idempotent — safe to re-run. Detects the GPU compute capability via
``nvidia-smi`` and installs from the matching index:

    sm_100 / sm_120  (Blackwell)   -> cu128
    sm_70 .. sm_90   (Volta..Hopper) -> default PyPI wheel is fine
    no GPU                          -> CPU build

Usage:  python scripts/install_torch.py [--dry-run]
"""

from __future__ import annotations

import shutil
import subprocess
import sys

# Blackwell needs cu128; bump here when a newer channel supersedes it.
_BLACKWELL_INDEX = "https://download.pytorch.org/whl/cu128"
_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def gpu_compute_cap() -> str | None:
    """Return the first GPU's compute capability (e.g. '10.0'), or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    caps = [line.strip() for line in out.splitlines() if line.strip()]
    return caps[0] if caps else None


def index_for(cap: str | None) -> str | None:
    """The pip --index-url for this GPU, or None to use the default PyPI wheel."""
    if cap is None:
        return _CPU_INDEX
    try:
        major = int(float(cap))
    except ValueError:
        return None
    if major >= 10:  # sm_100 (B200), sm_120 (RTX 50xx) and up — Blackwell+
        return _BLACKWELL_INDEX
    return None  # sm_70..sm_90: the default cu12x wheel already has these


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    cap = gpu_compute_cap()
    index = index_for(cap)
    where = f"index {index}" if index else "default PyPI wheel"
    print(f"GPU compute capability: {cap or 'none detected'}  ->  {where}")

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision"]
    if index:
        cmd += ["--index-url", index]
    print("running:", " ".join(cmd))
    if dry_run:
        return 0
    rc = subprocess.call(cmd)
    if rc == 0 and cap and int(float(cap)) >= 7:
        # Confirm CUDA actually works now (the whole point).
        check = (
            "import torch; "
            "x=torch.randn(8,8,device='cuda'); "
            "print('CUDA OK:', float((x@x).sum()) is not None)"
        )
        print("verifying CUDA...")
        subprocess.call([sys.executable, "-c", check])
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
