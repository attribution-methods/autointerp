"""Discovery harness — adapter between a candidate algorithm, a benchmark,
and a substrate evaluator.

Two execution paths:

- ``--dry-run``: deterministic stub — no model load, no benchmark, no
  registry lookup. Synthesizes AUC curves from the candidate's returned
  magnitudes so CI can round-trip the whole loop. Untouched by the
  substrate-aware refactor for backwards compatibility.

- Real path: the registry resolves ``(substrate, kind / decomposition)``
  to an evaluator callable; the benchmark resolves ``--benchmark`` (or
  the spec's ``DatasetSpec.generator_id``) to a pair generator and
  behavioral metric. The harness invokes the evaluator with both,
  receives the AUC payload, routes ``objective_value`` based on
  ``--objective``, and writes the JSON.

Output JSON shape is stable: ``mean_ablation_auc_k``,
``mean_steering_auc_k``, ``combined_auc_k``, ``patch_effect_recovery``
(when the evaluator emits it), ``top_features``, ``objective_metric``,
``objective_value``, ``algorithm``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------


def _load_algorithm(algorithm_path: Path):
    name = f"discovery_candidate_{algorithm_path.stem}"
    spec = importlib.util.spec_from_file_location(name, algorithm_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate algorithm at {algorithm_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "Algorithm"):
        instance = module.Algorithm()
        if not hasattr(instance, "score"):
            raise RuntimeError(
                f"{algorithm_path}: class Algorithm has no .score() method"
            )
        return instance.score
    if hasattr(module, "score"):
        return module.score
    raise RuntimeError(
        f"{algorithm_path}: must expose either `score(...)` or "
        "a class `Algorithm` with `.score(...)`"
    )


# ---------------------------------------------------------------------------
# Dry-run path (unchanged for CI)
# ---------------------------------------------------------------------------


def _dry_run(score_fn, *, top_k: int, k_grid: list[int]) -> dict[str, Any]:
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
            "per_pair": [],
        }
    magnitudes = sorted((abs(getattr(c, "score", 0.0)) for c in candidates), reverse=True)
    s = sum(magnitudes) or 1.0
    cum = []
    running = 0.0
    for m in magnitudes:
        running += m / s
        cum.append(min(running, 1.0))
    while len(cum) < len(k_grid):
        cum.append(cum[-1] if cum else 0.0)
    cum = cum[: len(k_grid)]
    abl_curves = [cum, cum]
    steer_curves = [
        [c * 0.9 for c in cum],
        [c * 0.85 for c in cum],
    ]
    from autointerp.pipelines.investigation.metrics import _mean_auc_k
    from autointerp.spec import MetricName

    abl = _mean_auc_k(
        {"delta_curve_per_pair": abl_curves, "k_grid": [float(k) for k in k_grid]},
        name=MetricName.MEAN_ABLATION_AUC_K,
    )
    steer = _mean_auc_k(
        {"delta_curve_per_pair": steer_curves, "k_grid": [float(k) for k in k_grid]},
        name=MetricName.MEAN_STEERING_AUC_K,
    )
    return {
        "mean_ablation_auc_k": abl,
        "mean_steering_auc_k": steer,
        "top_features": [
            {
                "layer": getattr(c, "layer", -1),
                "idx": getattr(c, "idx", getattr(c, "feature_id", -1)),
                "kind": getattr(c, "kind", "attn_head"),
                "score": float(getattr(c, "score", 0.0)),
                "token_pos": getattr(c, "token_pos", None),
            }
            for c in candidates[:top_k]
        ],
        "k_grid": k_grid,
        "per_pair": [
            {"ablation_curve": abl_curves[0], "steering_curve": steer_curves[0]},
            {"ablation_curve": abl_curves[1], "steering_curve": steer_curves[1]},
        ],
    }


# ---------------------------------------------------------------------------
# Real-path: registry resolution + benchmark dispatch
# ---------------------------------------------------------------------------


def _objective_from_payload(payload: dict[str, Any], objective: str) -> tuple[str, float]:
    """Route ``--objective`` onto the right field in the payload.

    Recognized objective values: ``ablation``, ``steering``, ``combined``,
    ``patch_effect_recovery``. The first three derive from AUC payload
    fields; the last reads ``payload["patch_effect_recovery"]`` directly
    (emitted by the components/attn_heads evaluator).
    """
    abl = float(payload.get("mean_ablation_auc_k", 0.0))
    steer = float(payload.get("mean_steering_auc_k", 0.0))
    if objective == "ablation":
        return "mean_ablation_auc_k", abl
    if objective == "steering":
        return "mean_steering_auc_k", steer
    if objective == "combined" or objective == "combined_auc_k":
        return "combined_auc_k", 0.5 * (abl + steer)
    if objective == "patch_effect_recovery":
        return "patch_effect_recovery", float(payload.get("patch_effect_recovery", 0.0))
    # Unknown but explicit — pass through if the evaluator emitted it,
    # otherwise default to combined.
    if objective in payload:
        try:
            return objective, float(payload[objective])
        except (TypeError, ValueError):
            pass
    return "combined_auc_k", 0.5 * (abl + steer)


def _build_pairs(
    *,
    benchmark_id: str,
    n_pairs: int,
    seed: int,
    split: str,
    model_id: str,
    device: str | None,
):
    """Resolve the benchmark module and generate pairs.

    Tokenizer comes from the substrate evaluator's model handle so the
    same benchmark works across GPT-2 / Llama / Qwen / Gemma. We load
    the model lazily here only because the benchmark needs the
    tokenizer to verify single-token names — once loaded it gets cached
    and the substrate evaluator's ``_get_model`` call is a no-op.
    """
    from autointerp.benchmarks import get_benchmark
    from autointerp.tools.model import load_model

    benchmark = get_benchmark(benchmark_id)
    model_handle = load_model(model_id, device=device)
    return (
        benchmark.generate_pairs(
            spec=None, n_pairs=n_pairs, seed=seed, tokenizer=model_handle.tokenizer, split=split,
        ),
        benchmark.behavioral_metric,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discovery harness")
    parser.add_argument("--algorithm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--k-values", default="1,5,10,20,50")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices. Empty = all layers.")
    parser.add_argument("--objective", default="combined")
    parser.add_argument("--substrate", default=None,
                        choices=[None, "components", "features"],
                        help="Discovery substrate. Required for non-dry-run paths.")
    parser.add_argument("--component-kind", default="attn_head",
                        help="Used when --substrate components.")
    parser.add_argument("--decomposition", default=None,
                        help="Used when --substrate features (e.g. sae_gemmascope).")
    parser.add_argument("--benchmark", default=None,
                        help="Benchmark id (matches DatasetSpec.generator_id).")
    parser.add_argument("--model-id", default=None,
                        help="HF model id for the substrate evaluator.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--evaluator", default=None,
                        help="Override module:attr. When set, skips registry lookup.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the deterministic stub evaluator (no model load).")
    args = parser.parse_args(argv)

    k_grid = [int(x) for x in args.k_values.split(",") if x.strip()]
    score_fn = _load_algorithm(args.algorithm)

    if args.dry_run:
        payload = _dry_run(score_fn, top_k=args.top_k, k_grid=k_grid)
    else:
        if args.substrate is None:
            raise SystemExit(
                "harness: --substrate is required for non-dry-run runs "
                "(omit it only with --dry-run)"
            )
        if args.benchmark is None:
            raise SystemExit("harness: --benchmark is required for non-dry-run runs")
        if args.model_id is None:
            raise SystemExit("harness: --model-id is required for non-dry-run runs")

        # Resolve the substrate evaluator.
        from autointerp.pipelines.investigation.discovery.evaluators import (
            get_evaluator,
        )
        evaluator = get_evaluator(
            substrate=args.substrate,
            component_kind=args.component_kind if args.substrate == "components" else None,
            decomposition=args.decomposition if args.substrate == "features" else None,
            override=args.evaluator,
        )

        # Resolve the benchmark + tokenized pairs.
        pairs, behavioral_metric = _build_pairs(
            benchmark_id=args.benchmark,
            n_pairs=args.n_pairs,
            seed=args.seed,
            split=args.split,
            model_id=args.model_id,
            device=args.device,
        )

        layers: list[int] | None = None
        if args.layers:
            layers = [int(x) for x in args.layers.split(",") if x.strip()]

        payload = evaluator(
            score_fn,
            model_id=args.model_id,
            device=args.device,
            pairs=pairs,
            behavioral_metric=behavioral_metric,
            layers=layers,
            top_k=args.top_k,
            k_grid=k_grid,
            context={
                "substrate": args.substrate,
                "component_kind": args.component_kind,
                "decomposition": args.decomposition,
                "benchmark": args.benchmark,
                "split": args.split,
            },
        )

    obj_name, obj_value = _objective_from_payload(payload, args.objective)
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
