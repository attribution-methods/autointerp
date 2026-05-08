#!/usr/bin/env python3
"""Small Jetson CUDA and cached GPT-2 preflight for the IOI run."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autointerp.tools.model import load_model


def main() -> int:
    print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 2
    print(f"device={torch.cuda.get_device_name(0)}")
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print(f"matmul=ok sample={float(c[0, 0]):.6f}")

    handle = load_model(
        "gpt2",
        device="cuda",
        dtype=torch.float16,
        local_files_only=True,
    )
    inputs = handle.tokenizer("Hello world", return_tensors="pt").to(handle.input_device())
    with torch.inference_mode():
        logits = handle.model(**inputs).logits
    print(f"gpt2_forward=ok shape={tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
