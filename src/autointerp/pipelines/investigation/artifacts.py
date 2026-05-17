"""Typed artifact gate for the investigation pipeline.

The agent does not write artifact JSON directly. It calls ``commit_artifact``,
which:

1. validates the payload against the matching schema in ``autointerp.schemas``;
2. resolves the on-disk path under the run directory;
3. refuses overwrites (artifacts are append-only);
4. stamps ``metadata["_provenance"]`` with stage_idx / split / committed_at;
5. for ``MetricResult`` payloads, requires a provenance token issued by
   ``compute_metric`` and verifies that the metric / value match the token;
6. atomically writes the file, updates ``state.json`` (artifact ref + token
   bookkeeping), and appends a ``log.jsonl`` entry.

State and log are written only via this gate (and its siblings). The agent
has no tool that touches them directly.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from autointerp import schemas as S

from .run_dir import RunHandle
from .state import RunState, StageStatus, now_iso, read_state, write_state

# kind name -> (schema class, id field, target subdir resolver)
# `subdir_kind` is one of: "prompt_batches", "activations", "generations", "findings".
ARTIFACT_KINDS: dict[str, tuple[type[BaseModel], str, str]] = {
    "PromptBatch": (S.PromptBatch, "batch_id", "prompt_batches"),
    "ActivationCacheRef": (S.ActivationCacheRef, "cache_id", "activations"),
    "GenerationSample": (S.GenerationSample, "sample_id", "generations"),
    "BehavioralFinding": (S.BehavioralFinding, "finding_id", "findings"),
    "CandidateSite": (S.CandidateSite, "site_id", "findings"),
    "FeatureFinding": (S.FeatureFinding, "feature_id", "findings"),
    "InterventionResult": (S.InterventionResult, "intervention_id", "findings"),
    "ValidationResult": (S.ValidationResult, "validation_id", "findings"),
    "MetricResult": (S.MetricResult, "metric_id", "findings"),
}

FINDING_PREFIX: dict[str, str] = {
    "BehavioralFinding": "behavioral",
    "CandidateSite": "candidate",
    "FeatureFinding": "feature",
    "InterventionResult": "intervention",
    "ValidationResult": "validation",
    "MetricResult": "metric",
}


class ArtifactGateError(Exception):
    """Raised when an artifact commit violates a gate invariant."""


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    artifact_id: str
    path: Path
    relpath: str
    stage_idx: int
    split: str


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".artifact.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _validate_payload(kind: str, payload: Any) -> BaseModel:
    if kind not in ARTIFACT_KINDS:
        raise ArtifactGateError(
            f"Unknown artifact kind {kind!r}. Valid: {sorted(ARTIFACT_KINDS)}"
        )
    cls, _id_field, _subdir = ARTIFACT_KINDS[kind]
    if isinstance(payload, cls):
        return payload
    if isinstance(payload, BaseModel):
        raise ArtifactGateError(
            f"payload is a {type(payload).__name__}; expected {cls.__name__} or a dict"
        )
    if not isinstance(payload, dict):
        raise ArtifactGateError("payload must be a dict or a typed schema instance")
    try:
        return cls.model_validate(payload)
    except ValidationError as exc:
        raise ArtifactGateError(f"{kind} payload failed validation:\n{exc}") from exc


def _artifact_id(kind: str, model: BaseModel) -> str:
    _cls, id_field, _subdir = ARTIFACT_KINDS[kind]
    raw = getattr(model, id_field)
    if raw is None:
        raise ArtifactGateError(f"{kind}.{id_field} is required and was empty")
    return str(raw)


def _resolve_path(handle: RunHandle, kind: str, artifact_id: str, stage_idx: int, stage_name: str) -> tuple[Path, str]:
    _cls, _id_field, subdir = ARTIFACT_KINDS[kind]
    safe_id = artifact_id.replace("/", "_")
    if subdir == "prompt_batches":
        path = handle.prompt_batches_dir / f"{safe_id}.json"
    elif subdir == "activations":
        path = handle.activations_dir / f"{safe_id}.json"
    elif subdir == "generations":
        path = handle.generations_dir / f"{safe_id}.json"
    elif subdir == "findings":
        prefix = FINDING_PREFIX[kind]
        stage_dir = handle.stage_findings_dir(stage_idx, stage_name)
        path = stage_dir / f"{prefix}_{safe_id}.json"
    else:
        raise ArtifactGateError(f"unhandled subdir {subdir!r}")
    relpath = path.relative_to(handle.root).as_posix()
    return path, relpath


def _resolve_split(kind: str, model: BaseModel, split_arg: str | None) -> str:
    if kind == "PromptBatch":
        # PromptBatch.split is the source of truth; an explicit arg must agree.
        batch_split = getattr(model, "split")
        if split_arg is not None and split_arg != batch_split:
            raise ArtifactGateError(
                f"split mismatch for PromptBatch: payload.split={batch_split!r} vs "
                f"split arg={split_arg!r}"
            )
        return batch_split
    if not split_arg:
        raise ArtifactGateError(
            f"split is required when committing {kind} (provenance tag for "
            "split-disjointness enforcement)"
        )
    return split_arg


def _stamp_provenance(model: BaseModel, *, stage_idx: int, split: str, committed_at: str) -> BaseModel:
    """Set metadata['_provenance']. Returns a new model copy (validate_assignment safe)."""
    md = dict(getattr(model, "metadata", {}) or {})
    if "_provenance" in md:
        raise ArtifactGateError(
            "metadata['_provenance'] is reserved; do not set it on the payload"
        )
    md["_provenance"] = {
        "stage_idx": stage_idx,
        "split": split,
        "committed_at": committed_at,
    }
    return model.model_copy(update={"metadata": md})


def _check_metric_provenance(
    state: RunState,
    payload: S.MetricResult,
    provenance_token: str | None,
    stage_idx: int,
) -> None:
    if not provenance_token:
        raise ArtifactGateError(
            "MetricResult requires a provenance_token issued by compute_metric"
        )
    record = state.pending_provenance_tokens.get(provenance_token)
    if record is None:
        raise ArtifactGateError(
            f"provenance_token {provenance_token!r} is unknown or already consumed"
        )
    if record.stage_idx != stage_idx:
        raise ArtifactGateError(
            f"provenance_token was issued at stage_idx={record.stage_idx} but "
            f"committing at stage_idx={stage_idx}"
        )
    if record.metric_id != payload.metric_id:
        raise ArtifactGateError(
            f"MetricResult.metric_id={payload.metric_id!r} does not match "
            f"token.metric_id={record.metric_id!r}"
        )
    if record.value != payload.value:
        raise ArtifactGateError(
            f"MetricResult.value={payload.value!r} does not match "
            f"token.value={record.value!r} — values may not be edited after compute_metric"
        )


def _append_log(handle: RunHandle, entry: dict[str, Any]) -> None:
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with handle.log_path.open("a") as fh:
        fh.write(line)


def _check_terminal(state: RunState) -> None:
    if state.terminal_state is not None:
        raise ArtifactGateError(
            f"run is in terminal state {state.terminal_state.value!r}; no further commits"
        )


def _current_stage(state: RunState, n_stages: int) -> tuple[int, str]:
    idx = state.current_stage_idx
    if idx < 0 or idx >= n_stages:
        raise ArtifactGateError(
            f"current_stage_idx={idx} is outside [0, {n_stages})"
        )
    rec = state.stage_status.get(str(idx))
    if rec is None:
        raise ArtifactGateError(f"missing stage_status[{idx}] in state.json")
    return idx, rec.stage


def commit_artifact(
    handle: RunHandle,
    kind: str,
    payload: Any,
    *,
    split: str | None = None,
    provenance_token: str | None = None,
) -> ArtifactRef:
    """Validate, place, and record a typed artifact for the current stage."""
    # Lazy import to break the circular: guards.py reads artifacts on disk.
    from .guards import GuardError, enforce_budget

    try:
        enforce_budget(handle)
    except GuardError as exc:
        raise ArtifactGateError(str(exc)) from exc
    state = read_state(handle.state_path)
    _check_terminal(state)

    model = _validate_payload(kind, payload)

    # Pull n_stages from the spec so we can bound the stage index.
    spec_data = json.loads(handle.spec_path.read_text())
    n_stages = len(spec_data.get("stages", []))
    stage_idx, stage_name = _current_stage(state, n_stages)

    artifact_id = _artifact_id(kind, model)
    resolved_split = _resolve_split(kind, model, split)
    path, relpath = _resolve_path(handle, kind, artifact_id, stage_idx, stage_name)
    if path.exists():
        raise ArtifactGateError(
            f"artifact already exists at {relpath}; artifacts are append-only"
        )

    if kind == "MetricResult":
        assert isinstance(model, S.MetricResult)
        # Flag B: when the provenance mechanism is off, MetricResult may be
        # committed with an agent-reported value and no token (number
        # fabrication is exactly the failure mode under test).
        if handle.flags.provenance_metrics:
            _check_metric_provenance(state, model, provenance_token, stage_idx)
    elif provenance_token is not None:
        raise ArtifactGateError(
            f"provenance_token is only valid for MetricResult; got {kind!r}"
        )

    committed_at = now_iso()
    stamped = _stamp_provenance(
        model, stage_idx=stage_idx, split=resolved_split, committed_at=committed_at
    )

    # Persist the artifact file.
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, stamped.model_dump_json(indent=2) + "\n")

    # Update state: append artifact ref; consume token if applicable.
    rec = state.stage_status[str(stage_idx)]
    rec.artifact_refs.append(relpath)
    if rec.status is StageStatus.PENDING:
        rec.status = StageStatus.IN_PROGRESS
        rec.started_at = rec.started_at or committed_at
    if kind == "MetricResult" and provenance_token and handle.flags.provenance_metrics:
        state.pending_provenance_tokens.pop(provenance_token, None)
        state.provenance_tokens_consumed += 1
    state.budget_consumed.tool_calls += 1
    write_state(handle.state_path, state)

    _append_log(
        handle,
        {
            "ts": committed_at,
            "stage_idx": stage_idx,
            "tool": "commit_artifact",
            "args": {"kind": kind, "artifact_id": artifact_id, "split": resolved_split},
            "ok": True,
            "result_summary": relpath,
            "budget_after": state.budget_consumed.model_dump(),
        },
    )

    ref = ArtifactRef(
        kind=kind,
        artifact_id=artifact_id,
        path=path,
        relpath=relpath,
        stage_idx=stage_idx,
        split=resolved_split,
    )

    if kind == "MetricResult":
        # Run abort_if predicates against the new MetricResult. If anything
        # trips, the run is now terminal — the agent will see this on its next
        # tool call.
        from .guards import check_abort_predicates

        check_abort_predicates(handle)

    return ref


__all__ = ["ArtifactGateError", "ArtifactRef", "commit_artifact", "ARTIFACT_KINDS"]
