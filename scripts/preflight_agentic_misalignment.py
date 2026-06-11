"""Pre-flight check for the agentic-misalignment golden run.

Verifies every dependency this investigation needs *before* the user
spends money on a Stage-0 LLM conversation that would only fail when the
agent tries to invoke petri.

Checks (each prints PASS / FAIL with what to fix):

  1. autointerp's [mechinterp] extra is importable (sae-lens, torch, hf-hub).
  2. circuitbreaker's vendored petri venv exists and imports cleanly.
  3. data/aoi.json exists in circuitbreaker and has the expected behavior.
  4. ANTHROPIC_API_KEY (or OPENROUTER_API_KEY) is set.
  5. HF_TOKEN is set (gated Gemma access).
  6. vLLM is reachable on the configured base URL and serves a Gemma model.
  7. autointerp's skill registry includes multi-turn-elicitation + pretrained-saes.

Usage:

  python3 scripts/preflight_agentic_misalignment.py [--circuitbreaker DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def _fail(msg: str, fix: str = "") -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")
    if fix:
        print(f"        fix: {fix}")


# ---- checks ---------------------------------------------------------------


def check_mechinterp() -> bool:
    print("\n[1] autointerp [mechinterp] extra")
    missing = []
    for mod in ("torch", "transformers", "huggingface_hub", "sae_lens", "requests"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        _fail(
            f"missing imports: {missing}",
            "pip install -e '.[mechinterp]' from the autointerp root",
        )
        return False
    _ok("torch, transformers, huggingface_hub, sae_lens, requests importable")
    return True


def check_petri_venv(circuitbreaker_root: Path) -> bool:
    print("\n[2] circuitbreaker petri venv")
    petri_python = circuitbreaker_root / ".venv-petri" / "bin" / "python"
    if not petri_python.exists():
        _fail(
            f"no petri python at {petri_python}",
            "follow circuitbreaker/behavioral_stage/README.md 'Petri install (patched)' section",
        )
        return False
    import subprocess

    res = subprocess.run(
        [str(petri_python), "-c",
         "import inspect_ai, inspect_petri; print('ok')"],
        capture_output=True, text=True, timeout=30,
    )
    if res.returncode != 0:
        _fail(
            f"petri venv import failed: {res.stderr.strip()[:200]}",
            "rebuild the venv per circuitbreaker/behavioral_stage/README.md",
        )
        return False
    _ok(f"{petri_python} imports inspect_ai + inspect_petri")
    return True


def check_aoi(circuitbreaker_root: Path, behavior: str) -> bool:
    print(f"\n[3] AOI seeds for behavior {behavior!r}")
    aoi_path = circuitbreaker_root / "behavioral_stage" / "data" / "aoi.json"
    if not aoi_path.exists():
        _fail(
            f"no aoi.json at {aoi_path}",
            "create the file with at least one entry for the chosen behavior",
        )
        return False
    try:
        data = json.loads(aoi_path.read_text())
    except json.JSONDecodeError as exc:
        _fail(f"aoi.json is malformed: {exc}", "fix the JSON syntax")
        return False
    if behavior not in data:
        _fail(
            f"aoi.json has no key {behavior!r}; available: {sorted(data)}",
            f"add an {behavior!r} key whose value is a list of seed-instruction strings",
        )
        return False
    seeds = data[behavior]
    if not isinstance(seeds, list) or not seeds:
        _fail(
            f"aoi.json[{behavior}] is not a non-empty list",
            "add at least one seed-instruction string",
        )
        return False
    _ok(f"aoi.json[{behavior!r}] has {len(seeds)} seed(s)")
    return True


def check_keys() -> bool:
    print("\n[4] auditor/judge API key")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openrouter"
        _ok(f"{provider} key set in environment")
        return True
    _fail(
        "neither ANTHROPIC_API_KEY nor OPENROUTER_API_KEY is set",
        "export one (and set the matching LLM_PROVIDER in circuitbreaker/behavioral_stage/.env)",
    )
    return False


def check_hf_token() -> bool:
    print("\n[5] HF_TOKEN (Gemma is gated)")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        _ok("HF token set in environment")
        return True
    _fail(
        "no HF_TOKEN / HUGGING_FACE_HUB_TOKEN in environment",
        "accept the Gemma 2 license at https://huggingface.co/google/gemma-2-9b-it, "
        "create a read token, export HF_TOKEN=hf_...",
    )
    return False


def check_vllm(base_url: str, expected_model: str) -> bool:
    print(f"\n[6] vLLM at {base_url}")
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            payload = json.loads(resp.read())
    except Exception as exc:
        _fail(
            f"can't reach {base_url}/models: {exc}",
            f"start vllm: `vllm serve {expected_model} --port 8000 "
            "--dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 8192`",
        )
        return False
    served = [m.get("id") for m in payload.get("data", [])]
    if expected_model not in served:
        _fail(
            f"vLLM is serving {served}, not {expected_model}",
            f"restart vllm with `vllm serve {expected_model} ...`",
        )
        return False
    _ok(f"vLLM serving {expected_model}")
    return True


def check_skills(autointerp_root: Path) -> bool:
    print("\n[7] autointerp skill registry")
    sys.path.insert(0, str(autointerp_root / "src"))
    from autointerp_agent.skills import SkillRegistry  # noqa: E402

    r = SkillRegistry.from_dir(autointerp_root / "skills")
    needed = {"multi-turn-elicitation", "pretrained-saes",
              "contrastive-directions", "causal-validation"}
    missing = needed - set(r.skills)
    if missing:
        _fail(
            f"missing skills: {sorted(missing)}",
            "ensure skills/<name>/SKILL.md exists and validates",
        )
        return False
    _ok(f"{len(needed)} required skills present")
    return True


# ---- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--circuitbreaker",
        default=str(here.parent / "circuitbreaker"),
        help="Path to circuitbreaker repo (default: ../circuitbreaker)",
    )
    parser.add_argument("--behavior", default="agentic_misalignment")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--target-model", default="google/gemma-2-9b-it")
    args = parser.parse_args()

    cb_root = Path(args.circuitbreaker).resolve()
    autointerp_root = here

    print(f"autointerp:    {autointerp_root}")
    print(f"circuitbreaker: {cb_root}")

    results = [
        check_mechinterp(),
        check_petri_venv(cb_root),
        check_aoi(cb_root, args.behavior),
        check_keys(),
        check_hf_token(),
        check_vllm(args.vllm_url, args.target_model),
        check_skills(autointerp_root),
    ]
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} checks passed.")
    if n_pass == n_total:
        print("\n\033[32mReady to run.\033[0m  Next:")
        print(
            "  autointerp investigate "
            "--model anthropic/claude-sonnet-4-6 "
            "--max-iterations 800 --max-revisions 2"
        )
        print("  Paste the prompt from scripts/agentic_misalignment_prompt.md")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
