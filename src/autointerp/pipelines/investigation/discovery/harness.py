"""Discovery harness — evaluate ONE candidate algorithm into a reward JSON.

Run as a subprocess by the loop so a candidate's import errors / exceptions
can't crash the parent run. Two paths:

- ``--dry-run``: deterministic stub. No model, no GPU. Synthesizes ablation /
  steering AUC curves from the candidate's returned magnitudes so the whole
  loop round-trips in tests.
- ``--evaluator module:attr``: real path. Resolves a user-supplied callable
  that loads the model, applies the candidate's ranking, ablates / steers the
  top-K over a K-sweep, and returns the AUC payload. Kept as a plug-in point so
  the minimal core ships without any heavyweight model dependency.

Output JSON is stable: ``mean_ablation_auc_k``, ``mean_steering_auc_k``,
``combined_auc_k``, ``objective_metric``, ``objective_value``, ``top_features``,
``algorithm``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_callable_from_file(path: Path, *, attr: str = "score"):
    name = f"discovery_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "Algorithm"):
        instance = module.Algorithm()
        if not hasattr(instance, attr):
            raise RuntimeError(f"{path}: Algorithm has no .{attr}() method")
        return getattr(instance, attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    raise RuntimeError(
        f"{path}: must expose `{attr}(...)` or a class `Algorithm` with `.{attr}(...)`"
    )


def _resolve_evaluator(ref: str):
    """Resolve a ``module:attr`` reference to a callable."""
    if ":" not in ref:
        raise SystemExit(f"--evaluator must be 'module:attr', got {ref!r}")
    mod_name, attr = ref.split(":", 1)
    module = importlib.import_module(mod_name)
    fn = getattr(module, attr, None)
    if fn is None:
        raise SystemExit(f"--evaluator {ref!r}: {attr} not found in {mod_name}")
    return fn


def _dry_run(score_fn, *, top_k: int, k_grid: list[int]) -> dict[str, Any]:
    """Deterministic stub: synthesize AUC curves from candidate magnitudes."""
    from autointerp.pipelines.investigation.metrics import _mean_auc_k
    from autointerp.spec import MetricName

    fake_pairs = [
        {"clean_input": "A", "clean_output": "x", "corrupted_output": "y"},
        {"clean_input": "B", "clean_output": "p", "corrupted_output": "q"},
    ]
    layers = [0, 1, 2]
    candidates = score_fn(
        fake_pairs,
        loader=None,
        layers=layers,
        device="cpu",
        top_k=top_k,
        context={"behavior": "dry_run", "metadata": {}},
    )
    if not candidates:
        return {
            "mean_ablation_auc_k": 0.0,
            "mean_steering_auc_k": 0.0,
            "top_features": [],
            "k_grid": k_grid,
        }
    magnitudes = sorted(
        (abs(float(getattr(c, "score", 0.0))) for c in candidates), reverse=True
    )
    total = sum(magnitudes) or 1.0
    cum: list[float] = []
    running = 0.0
    for m in magnitudes:
        running += m / total
        cum.append(min(running, 1.0))
    while len(cum) < len(k_grid):
        cum.append(cum[-1] if cum else 0.0)
    cum = cum[: len(k_grid)]
    abl_curves = [cum, cum]
    steer_curves = [[c * 0.9 for c in cum], [c * 0.85 for c in cum]]
    grid = [float(k) for k in k_grid]
    abl = _mean_auc_k(
        {"delta_curve_per_pair": abl_curves, "k_grid": grid},
        name=MetricName.MEAN_ABLATION_AUC_K,
    )
    steer = _mean_auc_k(
        {"delta_curve_per_pair": steer_curves, "k_grid": grid},
        name=MetricName.MEAN_STEERING_AUC_K,
    )
    return {
        "mean_ablation_auc_k": abl,
        "mean_steering_auc_k": steer,
        "top_features": [
            {
                "layer": getattr(c, "layer", -1),
                "idx": getattr(c, "idx", -1),
                "kind": getattr(c, "kind", "attn_head"),
                "score": float(getattr(c, "score", 0.0)),
                "token_pos": getattr(c, "token_pos", None),
            }
            for c in candidates[:top_k]
        ],
        "k_grid": k_grid,
    }


def _objective(payload: dict[str, Any], objective: str) -> tuple[str, float]:
    abl = float(payload.get("mean_ablation_auc_k", 0.0))
    steer = float(payload.get("mean_steering_auc_k", 0.0))
    if objective == "ablation":
        return "mean_ablation_auc_k", abl
    if objective == "steering":
        return "mean_steering_auc_k", steer
    # default: combined
    return "combined_auc_k", 0.5 * (abl + steer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discovery harness (one candidate)")
    parser.add_argument("--algorithm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--k-values", default="1,5,10,20,50")
    parser.add_argument("--objective", default="combined",
                        choices=["combined", "ablation", "steering"])
    parser.add_argument("--evaluator", default=None,
                        help="module:attr returning the AUC payload (real path).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Deterministic stub evaluator (no model load).")
    args = parser.parse_args(argv)

    k_grid = [int(x) for x in args.k_values.split(",") if x.strip()]
    score_fn = _load_callable_from_file(args.algorithm)

    if args.dry_run:
        payload = _dry_run(score_fn, top_k=args.top_k, k_grid=k_grid)
    elif args.evaluator:
        evaluator = _resolve_evaluator(args.evaluator)
        payload = evaluator(score_fn, top_k=args.top_k, k_grid=k_grid)
        if not isinstance(payload, dict):
            raise SystemExit("--evaluator must return a dict payload")
    else:
        raise SystemExit("harness: pass --dry-run or --evaluator module:attr")

    obj_name, obj_value = _objective(payload, args.objective)
    payload["combined_auc_k"] = 0.5 * (
        float(payload.get("mean_ablation_auc_k", 0.0))
        + float(payload.get("mean_steering_auc_k", 0.0))
    )
    payload["objective_metric"] = obj_name
    payload["objective_value"] = obj_value
    payload["algorithm"] = str(args.algorithm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"objective ({obj_name}): {obj_value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
