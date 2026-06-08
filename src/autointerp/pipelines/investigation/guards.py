"""Mechanically-evaluated abort predicates and budget enforcement.

Two responsibilities:

1. **Abort predicates.** ``spec.abort_if`` may carry typed ``AbortPredicate``
   entries (free-text strings are advisory only — see ``spec.py``). After any
   ``MetricResult`` is committed, ``check_abort_predicates`` walks the typed
   predicates, finds matching MetricResults (filtered by ``on_stage`` if
   set), and trips the run with ``terminal_state = ABORTED`` when any
   threshold is crossed.

2. **Budget enforcement.** ``enforce_budget`` is called at the entry of
   ``compute_metric`` and ``commit_artifact`` (via the tool wrappers) so that
   the *next* tool call is rejected when a cap would be exceeded. The
   middleware also flips the run to ``BUDGET_EXHAUSTED`` so subsequent calls
   short-circuit.

Both functions are pure on (state, spec, run-dir contents); they only mutate
state.json + log when they actually trip something.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autointerp.spec import (
    AbortPredicate,
    InvestigationSpec,
    InvestigationStage,
    typed_abort_predicates,
)

from .run_dir import RunHandle
from .state import (
    AbortRecord,
    RunState,
    TerminalState,
    now_iso,
    read_state,
    write_state,
)


class GuardError(Exception):
    pass


# ---- abort predicates ------------------------------------------------------


def _iter_metric_artifacts(handle: RunHandle) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if not handle.findings_dir.is_dir():
        return out
    for stage_dir in sorted(handle.findings_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        for path in sorted(stage_dir.glob("metric_*.json")):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            relpath = path.relative_to(handle.root).as_posix()
            out.append((relpath, payload))
    return out


def _apply_comparator(value: float, comparator: str, threshold: float) -> bool:
    if comparator == ">=":
        return value >= threshold
    if comparator == ">":
        return value > threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "==":
        return value == threshold
    raise GuardError(f"unknown comparator {comparator!r}")


def _stage_idx_for(spec: InvestigationSpec, stage: InvestigationStage) -> int | None:
    for i, s in enumerate(spec.stages):
        if s.stage == stage:
            return i
    return None


def _predicate_applies_to(
    predicate: AbortPredicate, payload_provenance_stage_idx: int, spec: InvestigationSpec
) -> bool:
    if predicate.on_stage is None:
        return True
    target_idx = _stage_idx_for(spec, predicate.on_stage)
    return target_idx is not None and target_idx == payload_provenance_stage_idx


def _find_first_trip(
    spec: InvestigationSpec, handle: RunHandle
) -> tuple[AbortPredicate, str, float] | None:
    """Return (predicate, metric_result_ref, observed_value) for the first trip,
    or None if nothing trips. Stable ordering: predicates in spec order, then
    artifacts in path order — first match wins."""
    predicates = typed_abort_predicates(spec)
    if not predicates:
        return None
    artifacts = _iter_metric_artifacts(handle)
    for pred in predicates:
        for relpath, payload in artifacts:
            md = payload.get("metadata") or {}
            if md.get("metric_name") != pred.metric.value:
                continue
            prov = md.get("_provenance") or {}
            stage_idx = prov.get("stage_idx")
            if not isinstance(stage_idx, int):
                continue
            if not _predicate_applies_to(pred, stage_idx, spec):
                continue
            value = float(payload.get("value", float("nan")))
            if _apply_comparator(value, pred.comparator, pred.threshold):
                return pred, relpath, value
    return None


def check_abort_predicates(handle: RunHandle) -> AbortRecord | None:
    """Scan committed MetricResults; trip the run if any AbortPredicate fires.

    Idempotent. Returns the AbortRecord if (any) predicate is currently
    tripped — even one previously recorded — and None otherwise.
    """
    state = read_state(handle.state_path)
    if state.abort_triggered is not None:
        return state.abort_triggered
    if state.terminal_state is not None:
        return None

    spec = InvestigationSpec.model_validate_json(handle.spec_path.read_text())
    trip = _find_first_trip(spec, handle)
    if trip is None:
        return None

    pred, relpath, value = trip
    record = AbortRecord(
        predicate_id=pred.predicate_id,
        metric=pred.metric.value,
        value=value,
        at_stage_idx=state.current_stage_idx,
        at=now_iso(),
    )
    state.abort_triggered = record
    state.terminal_state = TerminalState.ABORTED
    state.run_ended_at = record.at
    write_state(handle.state_path, state)

    line = (
        json.dumps(
            {
                "ts": record.at,
                "stage_idx": state.current_stage_idx,
                "tool": "abort_predicate",
                "args": {
                    "predicate_id": pred.predicate_id,
                    "metric": pred.metric.value,
                    "metric_result_ref": relpath,
                },
                "ok": True,
                "result_summary": (
                    f"trip: value={value!r} {pred.comparator} {pred.threshold!r}"
                ),
                "budget_after": state.budget_consumed.model_dump(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    with handle.log_path.open("a") as fh:
        fh.write(line)
    return record


# ---- budget ----------------------------------------------------------------


def _budget_exceeded_reason(state: RunState, spec: InvestigationSpec) -> str | None:
    bc = state.budget_consumed
    b = spec.budget
    if b.max_tool_calls is not None and bc.tool_calls >= b.max_tool_calls:
        return f"max_tool_calls={b.max_tool_calls} reached"
    if b.max_samples is not None and bc.samples >= b.max_samples:
        return f"max_samples={b.max_samples} reached"
    if b.max_gpu_seconds is not None and bc.gpu_seconds >= b.max_gpu_seconds:
        return f"max_gpu_seconds={b.max_gpu_seconds} reached"
    if b.max_wallclock_seconds is not None and bc.wallclock_seconds >= b.max_wallclock_seconds:
        return f"max_wallclock_seconds={b.max_wallclock_seconds} reached"
    return None


def enforce_budget(handle: RunHandle) -> None:
    """Refuse the next tool call if the spec budget is exhausted.

    Called at the entry of compute_metric and commit_artifact via the tool
    wrappers. Trips the run to ``BUDGET_EXHAUSTED`` so subsequent calls bail
    out cleanly.
    """
    state = read_state(handle.state_path)
    if state.terminal_state is TerminalState.BUDGET_EXHAUSTED:
        raise GuardError("budget exhausted")
    if state.terminal_state is not None:
        return  # already terminal for some other reason; let that gate handle it

    spec = InvestigationSpec.model_validate_json(handle.spec_path.read_text())
    reason = _budget_exceeded_reason(state, spec)
    if reason is None:
        return

    state.terminal_state = TerminalState.BUDGET_EXHAUSTED
    state.run_ended_at = now_iso()
    write_state(handle.state_path, state)
    line = (
        json.dumps(
            {
                "ts": state.run_ended_at,
                "stage_idx": state.current_stage_idx,
                "tool": "enforce_budget",
                "args": {},
                "ok": False,
                "result_summary": reason,
                "budget_after": state.budget_consumed.model_dump(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    with handle.log_path.open("a") as fh:
        fh.write(line)
    raise GuardError(f"budget exhausted: {reason}")


__all__ = ["GuardError", "check_abort_predicates", "enforce_budget"]


_ = (Path,)
