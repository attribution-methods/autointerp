"""One-shot success-criterion evaluator.

The investigation pipeline pre-registers ``success_criteria`` as part of the
spec. This module is the only place those criteria are scored. Two invariants
that make pre-registration meaningful live here:

1. **One-shot per spec rev.** Once ``evaluate_criterion(criterion_id)`` runs
   for a given spec revision, the result is frozen in
   ``state.criteria_evaluated`` and re-calls return the cached record.
2. **Split disjointness.** A ``Criterion`` declares an ``on_split`` (typically
   ``heldout`` for the headline claim). The evaluator looks up matching
   ``MetricResult`` artifacts whose ``_provenance.split`` matches; results
   tagged with another split are refused, blocking discovery/validation
   contamination.

Failed criteria flip the run to ``terminal_state = CRITERION_FAILED``. The
only legitimate way forward from there is a child spec via
``request_spec_revision`` (in ``tools.py``).

A criterion may also be evaluated ``INCONCLUSIVE``: the agent passes an
``inconclusive_reason`` when the metric was computed but a stated limitation
(too few samples, wide CI, degenerate data) prevents reading a verdict off
the value. INCONCLUSIVE is one-shot like PASS/FAIL and counts as "evaluated"
for the final ``advance_stage`` gate, but it is terminal-state-neutral — it
does not flip the run to ``CRITERION_FAILED``. This lets a run record an
honest non-result instead of forcing a binary it cannot defend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autointerp.spec import Criterion, InvestigationSpec, MetricName

from .run_dir import RunHandle
from .state import (
    CriterionRecord,
    RunState,
    TerminalState,
    Verdict,
    now_iso,
    read_state,
    write_state,
)


class CriterionGateError(Exception):
    pass


def _load_spec(handle: RunHandle) -> InvestigationSpec:
    return InvestigationSpec.model_validate_json(handle.spec_path.read_text())


def _find_criterion(spec: InvestigationSpec, criterion_id: str) -> Criterion:
    for c in spec.success_criteria:
        if c.criterion_id == criterion_id:
            return c
    valid = [c.criterion_id for c in spec.success_criteria]
    raise CriterionGateError(
        f"unknown criterion_id {criterion_id!r}; valid: {valid}"
    )


def _iter_metric_artifacts(handle: RunHandle) -> list[tuple[Path, dict[str, Any]]]:
    """Walk all stage findings dirs and return (path, payload) for every metric_*.json."""
    out: list[tuple[Path, dict[str, Any]]] = []
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
            out.append((path, payload))
    return out


def _matches(
    payload: dict[str, Any],
    *,
    metric: MetricName,
    split: str,
    custom_name: str | None = None,
    enforce_split: bool = True,
) -> bool:
    md = payload.get("metadata") or {}
    expected = custom_name if metric == MetricName.CUSTOM else metric.value
    if md.get("metric_name") != expected:
        return False
    if not enforce_split:
        # Flag C off: split tags are still written but not used to match —
        # discovery-set metrics may satisfy a heldout criterion.
        return True
    prov = md.get("_provenance") or {}
    return prov.get("split") == split


def _resolve_metric_result(
    handle: RunHandle,
    criterion: Criterion,
    metric_result_ref: str | None,
) -> tuple[Path, dict[str, Any]]:
    """Pick the single MetricResult artifact that scores ``criterion``.

    If the agent passed an explicit ``metric_result_ref`` (path under run
    root), use it after verifying metric/split match. Otherwise scan all
    committed metric artifacts; require exactly one match.
    """
    if metric_result_ref:
        path = handle.root / metric_result_ref
        if not path.exists():
            raise CriterionGateError(
                f"metric_result_ref {metric_result_ref!r} does not exist under run root"
            )
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise CriterionGateError(
                f"metric_result_ref {metric_result_ref!r} is not valid JSON: {exc}"
            ) from exc
        md = payload.get("metadata") or {}
        observed_metric = md.get("metric_name")
        observed_split = (md.get("_provenance") or {}).get("split")
        if criterion.metric == MetricName.CUSTOM:
            expected_metric = (
                criterion.custom_metric_def.name
                if criterion.custom_metric_def is not None
                else None
            )
        else:
            expected_metric = criterion.metric.value
        if observed_metric != expected_metric:
            raise CriterionGateError(
                f"metric_result_ref measures {observed_metric!r} but "
                f"criterion {criterion.criterion_id!r} requires "
                f"{expected_metric!r}"
            )
        if handle.flags.split_disjoint and observed_split != criterion.on_split:
            raise CriterionGateError(
                f"split disjointness violated: metric_result_ref is tagged "
                f"split={observed_split!r} but criterion requires on_split="
                f"{criterion.on_split!r}"
            )
        return path, payload

    # Auto-discover.
    custom_name = (
        criterion.custom_metric_def.name
        if criterion.metric == MetricName.CUSTOM and criterion.custom_metric_def is not None
        else None
    )
    candidates = [
        (p, pl)
        for p, pl in _iter_metric_artifacts(handle)
        if _matches(
            pl,
            metric=criterion.metric,
            split=criterion.on_split,
            custom_name=custom_name,
            enforce_split=handle.flags.split_disjoint,
        )
    ]
    if not candidates:
        raise CriterionGateError(
            f"no committed MetricResult matches criterion {criterion.criterion_id!r} "
            f"(metric={criterion.metric.value!r}, on_split={criterion.on_split!r}). "
            "Compute and commit one before evaluating."
        )
    if len(candidates) > 1:
        relpaths = sorted(p.relative_to(handle.root).as_posix() for p, _ in candidates)
        raise CriterionGateError(
            f"multiple MetricResults match criterion {criterion.criterion_id!r}; "
            f"pass an explicit metric_result_ref. Candidates: {relpaths}"
        )
    return candidates[0]


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
    raise CriterionGateError(f"unknown comparator {comparator!r}")


def _append_log(handle: RunHandle, entry: dict[str, Any]) -> None:
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with handle.log_path.open("a") as fh:
        fh.write(line)


def evaluate_criterion(
    handle: RunHandle,
    criterion_id: str,
    *,
    metric_result_ref: str | None = None,
    inconclusive_reason: str | None = None,
) -> CriterionRecord:
    """Evaluate one ``Criterion`` exactly once per spec rev.

    When ``inconclusive_reason`` is a non-empty string the criterion is
    recorded ``INCONCLUSIVE``: the matching MetricResult is still resolved and
    its value recorded for the report, but the threshold comparison is skipped
    and the run is *not* flipped to ``CRITERION_FAILED``. Use this for honest
    non-results (too few samples, wide CI, degenerate data) rather than
    forcing a PASS/FAIL the evidence cannot support.
    """
    state = read_state(handle.state_path)
    if (
        state.terminal_state is not None
        and state.terminal_state is not TerminalState.CRITERION_FAILED
    ):
        # CRITERION_FAILED is the one terminal state where re-evaluating cached
        # results is still meaningful (so the agent can read what failed). All
        # other terminal states refuse new evaluations.
        raise CriterionGateError(
            f"run is in terminal state {state.terminal_state.value!r}; "
            "no further criterion evaluations"
        )

    spec = _load_spec(handle)
    criterion = _find_criterion(spec, criterion_id)

    cached = state.criteria_evaluated.get(criterion_id)
    if cached is not None:
        return cached

    metric_path, payload = _resolve_metric_result(handle, criterion, metric_result_ref)
    if "value" not in payload:
        raise CriterionGateError(
            f"MetricResult at {metric_path} is missing a 'value' field"
        )
    value = float(payload["value"])

    reason = inconclusive_reason.strip() if isinstance(inconclusive_reason, str) else None
    if reason:
        verdict = Verdict.INCONCLUSIVE
    else:
        reason = None
        verdict = (
            Verdict.PASS
            if _apply_comparator(value, criterion.comparator, criterion.threshold)
            else Verdict.FAIL
        )

    record = CriterionRecord(
        verdict=verdict,
        value=value,
        metric=criterion.metric.value,
        comparator=criterion.comparator,
        threshold=criterion.threshold,
        metric_result_ref=metric_path.relative_to(handle.root).as_posix(),
        evaluated_at=now_iso(),
        inconclusive_reason=reason,
    )

    state.criteria_evaluated[criterion_id] = record
    state.budget_consumed.tool_calls += 1
    if verdict is Verdict.FAIL and state.terminal_state is None:
        state.terminal_state = TerminalState.CRITERION_FAILED
        state.run_ended_at = record.evaluated_at
    write_state(handle.state_path, state)

    if verdict is Verdict.INCONCLUSIVE:
        result_summary = (
            f"value={value!r} -> INCONCLUSIVE ({reason})"
        )
    else:
        result_summary = (
            f"value={value!r} {criterion.comparator} {criterion.threshold!r} "
            f"-> {verdict.value}"
        )
    _append_log(
        handle,
        {
            "ts": record.evaluated_at,
            "stage_idx": state.current_stage_idx,
            "tool": "evaluate_criterion",
            "args": {
                "criterion_id": criterion_id,
                "metric_result_ref": record.metric_result_ref,
                "inconclusive_reason": reason,
            },
            "ok": True,
            "result_summary": result_summary,
            "budget_after": state.budget_consumed.model_dump(),
        },
    )
    from .progress import refresh_progress

    refresh_progress(handle)
    return record


__all__ = ["CriterionGateError", "evaluate_criterion"]


# Keep RunState in scope for type-checkers / future hooks.
_ = RunState
