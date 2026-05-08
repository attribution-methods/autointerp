"""Deterministic Stage 0 bridge for the IOI GPT-2-small investigation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from autointerp.pipelines.ioi_stage0 import run_ioi_stage0_forward, write_metric_inputs

from .artifacts import ArtifactRef, commit_artifact
from .criteria import evaluate_criterion
from .main import prepare_run
from .metrics import compute_metric
from .stages import advance_stage, current_stage_view
from .state import read_state


DEFAULT_STAGE0_OUTPUT = "scratch/ioi_stage0_metric_inputs_compact.json"
PROMPT_BATCH_ID = "ioi_stage0_dev_pairs"


def _load_metric_inputs(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"metric input file not found: {path}") from exc
    required = {"compact_accuracy_inputs", "compact_logit_diff_inputs", "summary"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"metric input file {path} is missing keys: {missing}")
    return payload


def _metric_is_committed(handle: Any, metric_name: str) -> bool:
    view = current_stage_view(handle)
    committed = set(view.get("committed_metric_names") or [])
    return metric_name in committed


def _compute_and_commit(
    handle: Any,
    *,
    metric: str,
    metric_id: str,
    inputs: dict[str, Any],
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
        metadata={"source": "ioi_stage0_bridge"},
    )
    return commit_artifact(
        handle,
        "MetricResult",
        payload,
        split="dev",
        provenance_token=token,
    )


def _commit_prompt_batch_if_available(
    handle: Any,
    metric_payload: dict[str, Any],
) -> ArtifactRef | None:
    records = metric_payload.get("prompt_records")
    if not isinstance(records, list) or not records:
        return None
    if (handle.prompt_batches_dir / f"{PROMPT_BATCH_ID}.json").exists():
        return None

    cases = []
    for idx, record in enumerate(records):
        cases.append(
            {
                "prompt_id": f"ioi-dev-{idx:04d}",
                "messages": [{"role": "user", "content": record["prompt"]}],
                "expected_behavior": record["io"],
                "tags": ["ioi", record["template"].lower()],
                "metadata": {
                    "template": record["template"],
                    "corrupted_prompt": record["corrupted_prompt"],
                    "io": record["io"],
                    "s": record["s"],
                    "A": record["A"],
                    "B": record["B"],
                    "C": record["C"],
                    "place": record["place"],
                    "object": record["object"],
                    "token_selector": "final",
                },
            }
        )

    return commit_artifact(
        handle,
        "PromptBatch",
        {
            "batch_id": PROMPT_BATCH_ID,
            "behavior_id": "ioi-io-over-s-logit",
            "split": "dev",
            "cases": cases,
            "metadata": {
                "source": "ioi_stage0_bridge",
                "description": "Clean IOI prompts with paired ABC corrupted prompts in case metadata.",
            },
        },
    )


def _ensure_metric_inputs(
    handle: Any,
    *,
    metrics_json: Path | None,
    include_raw: bool,
    allow_download: bool,
    batch_size: int,
) -> tuple[dict[str, Any], Path]:
    if metrics_json is not None:
        return _load_metric_inputs(metrics_json), metrics_json

    payload = run_ioi_stage0_forward(
        device="cuda",
        dtype=torch.float16,
        batch_size=batch_size,
        local_files_only=not allow_download,
        include_raw=include_raw,
        include_prompt_records=True,
    )
    out_path = handle.root / DEFAULT_STAGE0_OUTPUT
    write_metric_inputs(out_path, payload)
    return payload, out_path


def bridge_stage0(
    *,
    spec: str | Path,
    runs_root: str | Path,
    resume: bool | None = None,
    metrics_json: str | Path | None = None,
    min_accuracy: float = 0.95,
    min_logit_diff: float = 2.0,
    include_raw: bool = False,
    allow_download: bool = False,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Commit Stage 0 IOI metrics and advance the run to Stage 1.

    By default this performs the GPU forward pass. Tests and recovery flows can
    pass ``metrics_json`` containing compact metric inputs to skip recomputing.
    """
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
    if state.current_stage_idx > 0:
        return {
            "run_root": handle.root.as_posix(),
            "status": "already_advanced",
            "current_stage_idx": state.current_stage_idx,
        }

    metric_payload, metric_input_path = _ensure_metric_inputs(
        handle,
        metrics_json=Path(metrics_json) if metrics_json is not None else None,
        include_raw=include_raw,
        allow_download=allow_download,
        batch_size=batch_size,
    )
    summary = metric_payload["summary"]
    accuracy = float(summary["accuracy"])
    mean_logit_diff = float(summary["mean_logit_diff"])
    if accuracy < min_accuracy:
        raise RuntimeError(
            f"Stage 0 accuracy {accuracy:.4f} is below required {min_accuracy:.4f}"
        )
    if mean_logit_diff < min_logit_diff:
        raise RuntimeError(
            "Stage 0 mean_logit_diff "
            f"{mean_logit_diff:.4f} is below required {min_logit_diff:.4f}"
        )

    refs: list[str] = []
    prompt_ref = _commit_prompt_batch_if_available(handle, metric_payload)
    if prompt_ref is not None:
        refs.append(prompt_ref.relpath)

    if not _metric_is_committed(handle, "accuracy"):
        ref = _compute_and_commit(
            handle,
            metric="accuracy",
            metric_id="stage0_accuracy",
            inputs=metric_payload["compact_accuracy_inputs"],
            threshold=min_accuracy,
            comparator=">=",
        )
        refs.append(ref.relpath)
    if not _metric_is_committed(handle, "logit_diff"):
        ref = _compute_and_commit(
            handle,
            metric="logit_diff",
            metric_id="stage0_logit_diff",
            inputs=metric_payload["compact_logit_diff_inputs"],
        )
        refs.append(ref.relpath)

    state = read_state(handle.state_path)
    criterion = state.criteria_evaluated.get("behavioral-sanity")
    if criterion is None:
        criterion = evaluate_criterion(handle, "behavioral-sanity")
    if not criterion.passed:
        raise RuntimeError("behavioral-sanity criterion failed")

    state = read_state(handle.state_path)
    advanced: dict[str, Any] | None = None
    if state.current_stage_idx == 0:
        advanced = advance_stage(handle)

    final_state = read_state(handle.state_path)
    return {
        "run_root": handle.root.as_posix(),
        "status": "advanced" if advanced else "stage0_ready",
        "metric_input_path": metric_input_path.as_posix(),
        "accuracy": accuracy,
        "mean_logit_diff": mean_logit_diff,
        "artifact_refs": refs,
        "criterion": criterion.model_dump(),
        "advance": advanced,
        "current_stage_idx": final_state.current_stage_idx,
        "terminal_state": (
            final_state.terminal_state.value if final_state.terminal_state else None
        ),
    }


__all__ = ["DEFAULT_STAGE0_OUTPUT", "PROMPT_BATCH_ID", "bridge_stage0"]
