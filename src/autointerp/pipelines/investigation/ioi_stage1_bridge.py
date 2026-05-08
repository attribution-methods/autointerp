"""Deterministic Stage 1 localization bridge for IOI head patching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from autointerp.tools.model import load_model
from autointerp.tools.patching import patch_head_group_metric, sweep_head_patch_recovery

from .artifacts import ArtifactRef, commit_artifact
from .criteria import evaluate_criterion
from .main import prepare_run
from .metrics import compute_metric
from .stages import advance_stage, current_stage_view
from .state import read_state


DEFAULT_STAGE1_OUTPUT = "scratch/ioi_stage1_head_patch_results.json"


def _load_prompt_pairs(handle: Any, *, max_samples: int | None) -> tuple[list[str], list[str], list[str], list[str]]:
    path = handle.prompt_batches_dir / "ioi_stage0_dev_pairs.json"
    payload = json.loads(path.read_text())
    cases = payload["cases"]
    if max_samples is not None:
        cases = cases[:max_samples]
    clean = [case["messages"][0]["content"] for case in cases]
    corrupt = [case["metadata"]["corrupted_prompt"] for case in cases]
    io_tokens = [case["metadata"]["io"] for case in cases]
    s_tokens = [case["metadata"]["s"] for case in cases]
    return clean, corrupt, io_tokens, s_tokens


def _load_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    required = {
        "clean_metric",
        "corrupt_metric",
        "top_k_patched_metric",
        "top_k_kl_per_sample",
        "top_k_heads",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Stage 1 results file {path} is missing keys: {missing}")
    return payload


def _metric_is_committed(handle: Any, metric_name: str) -> bool:
    view = current_stage_view(handle)
    committed = set(view.get("committed_metric_names") or [])
    return metric_name in committed


def _candidate_path_exists(handle: Any, site_id: str) -> bool:
    state = read_state(handle.state_path)
    rec = state.stage_status[str(state.current_stage_idx)]
    stage_dir = handle.stage_findings_dir(state.current_stage_idx, rec.stage)
    return (stage_dir / f"candidate_{site_id}.json").exists()


def _commit_candidate_sites(handle: Any, top_heads: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for rank, site in enumerate(top_heads, start=1):
        layer = int(site["layer"])
        head = int(site["head"])
        site_id = f"stage1_L{layer}H{head}"
        if _candidate_path_exists(handle, site_id):
            continue
        ref = commit_artifact(
            handle,
            "CandidateSite",
            {
                "site_id": site_id,
                "source": "gpt2.attn.c_proj_input",
                "method": "clean_to_corrupt_final_token_head_patch",
                "score": float(site["recovery"]),
                "layer": layer,
                "component": "attn",
                "head_index": head,
                "token_index": int(site.get("token_index", -1)),
                "metadata": {
                    "rank": rank,
                    "patched_metric": float(site["patched_metric"]),
                    "metric": "patch_effect_recovery",
                },
            },
            split="dev",
        )
        refs.append(ref.relpath)
    return refs


def _compute_and_commit_metric(
    handle: Any,
    *,
    metric: str,
    metric_id: str,
    inputs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    threshold: float | None = None,
    comparator: str | None = None,
) -> ArtifactRef:
    payload, token = compute_metric(
        handle,
        metric=metric,
        metric_id=metric_id,
        inputs=inputs,
        threshold=threshold,
        comparator=comparator,
        metadata={"source": "ioi_stage1_bridge", **(metadata or {})},
    )
    return commit_artifact(
        handle,
        "MetricResult",
        payload,
        split="dev",
        provenance_token=token,
    )


def _run_stage1_sweep(
    handle: Any,
    *,
    top_k: int,
    max_samples: int | None,
    batch_size: int,
    allow_download: bool,
) -> tuple[dict[str, Any], Path]:
    clean, corrupt, io_tokens, s_tokens = _load_prompt_pairs(handle, max_samples=max_samples)
    model = load_model(
        "gpt2",
        device="cuda",
        dtype=torch.float16,
        local_files_only=not allow_download,
    )
    sweep = sweep_head_patch_recovery(
        model,
        clean,
        corrupt,
        io_tokens,
        s_tokens,
        batch_size=batch_size,
        token_index=-1,
    )
    top_heads = sweep["top_heads"][:top_k]
    group = patch_head_group_metric(
        model,
        clean,
        corrupt,
        io_tokens,
        s_tokens,
        top_heads,
        batch_size=batch_size,
        token_index=-1,
    )
    denom = float(sweep["clean_metric"]) - float(sweep["corrupt_metric"])
    top_k_recovery = (
        (float(group["patched_metric"]) - float(sweep["corrupt_metric"])) / denom
        if abs(denom) >= 1e-9
        else float("nan")
    )
    result = {
        **sweep,
        "top_k": top_k,
        "top_k_heads": top_heads,
        "top_k_patched_metric": float(group["patched_metric"]),
        "top_k_recovery": float(top_k_recovery),
        "top_k_kl_per_sample": group["kl_per_sample"],
        "max_samples": max_samples,
        "batch_size": batch_size,
    }
    out_path = handle.root / DEFAULT_STAGE1_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result))
    return result, out_path


def bridge_stage1(
    *,
    spec: str | Path,
    runs_root: str | Path,
    resume: bool | None = True,
    results_json: str | Path | None = None,
    top_k: int = 10,
    max_samples: int | None = None,
    batch_size: int = 8,
    allow_download: bool = False,
    evaluate_localization: bool = True,
) -> dict[str, Any]:
    """Run or load Stage 1 IOI head patching, commit metrics, and advance."""
    handle, _spec_obj, state = prepare_run(
        spec,
        runs_root=Path(runs_root),
        resume=resume,
    )
    if state.terminal_state is not None:
        return {
            "run_root": handle.root.as_posix(),
            "status": "terminal",
            "terminal_state": state.terminal_state.value,
            "current_stage_idx": state.current_stage_idx,
        }
    if state.current_stage_idx > 1:
        return {
            "run_root": handle.root.as_posix(),
            "status": "already_advanced",
            "current_stage_idx": state.current_stage_idx,
        }
    if state.current_stage_idx != 1:
        raise RuntimeError(f"Stage 1 bridge requires current_stage_idx=1, got {state.current_stage_idx}")

    if results_json is not None:
        results_path = Path(results_json)
        results = _load_results(results_path)
    else:
        results, results_path = _run_stage1_sweep(
            handle,
            top_k=top_k,
            max_samples=max_samples,
            batch_size=batch_size,
            allow_download=allow_download,
        )

    refs = _commit_candidate_sites(handle, list(results["top_k_heads"]))

    if not _metric_is_committed(handle, "kl_to_clean"):
        ref = _compute_and_commit_metric(
            handle,
            metric="kl_to_clean",
            metric_id="stage1_kl_to_clean",
            inputs={"kl_per_sample": results["top_k_kl_per_sample"]},
            metadata={
                "top_k": int(results.get("top_k", len(results["top_k_heads"]))),
                "n_samples": int(results.get("n_samples", len(results["top_k_kl_per_sample"]))),
            },
        )
        refs.append(ref.relpath)

    if not _metric_is_committed(handle, "patch_effect_recovery"):
        ref = _compute_and_commit_metric(
            handle,
            metric="patch_effect_recovery",
            metric_id="stage1_patch_effect_recovery",
            inputs={
                "clean_metric": results["clean_metric"],
                "corrupt_metric": results["corrupt_metric"],
                "patched_metric": results["top_k_patched_metric"],
            },
            threshold=0.85,
            comparator=">=",
            metadata={
                "top_k": int(results.get("top_k", len(results["top_k_heads"]))),
                "top_heads": results["top_k_heads"],
                "results_ref": results_path.relative_to(handle.root).as_posix()
                if results_path.is_relative_to(handle.root)
                else results_path.as_posix(),
            },
        )
        refs.append(ref.relpath)

    criterion = None
    if evaluate_localization:
        criterion = evaluate_criterion(handle, "localization-recovery")
        if not criterion.passed:
            raise RuntimeError("localization-recovery criterion failed")

    state = read_state(handle.state_path)
    advanced = None
    if state.current_stage_idx == 1 and state.terminal_state is None:
        advanced = advance_stage(handle)

    final_state = read_state(handle.state_path)
    return {
        "run_root": handle.root.as_posix(),
        "status": "advanced" if advanced else "stage1_ready",
        "results_path": results_path.as_posix(),
        "clean_metric": float(results["clean_metric"]),
        "corrupt_metric": float(results["corrupt_metric"]),
        "top_k_patched_metric": float(results["top_k_patched_metric"]),
        "top_k_recovery": float(results.get("top_k_recovery", 0.0)),
        "top_k_heads": results["top_k_heads"],
        "artifact_refs": refs,
        "criterion": criterion.model_dump() if criterion is not None else None,
        "advance": advanced,
        "current_stage_idx": final_state.current_stage_idx,
        "terminal_state": (
            final_state.terminal_state.value if final_state.terminal_state else None
        ),
    }


__all__ = ["DEFAULT_STAGE1_OUTPUT", "bridge_stage1"]
