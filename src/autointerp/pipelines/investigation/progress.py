"""Agent-readable progress digest for an investigation run.

The run directory already holds the authoritative state (``state.json``), an
append-only audit (``log.jsonl``), and a full transcript
(``assistant_turns.jsonl`` / ``tool_invocations.jsonl``). None of those are
something the agent should re-read every turn — they are big and grow without
bound.

``progress.md`` is the compact, regenerated-in-place sibling: stage checklist,
per-criterion verdicts (including INCONCLUSIVE), budget, and any abort. It is
rewritten after every state-mutating gate (``advance_stage``,
``evaluate_criterion``) and on run init, so a single cheap ``read_file
progress.md`` (or the ``get_progress`` tool) re-grounds the agent after a long
stretch of bash work without paying for the transcript.

Pure render + atomic write; never raises into a gate (a digest failure must
not break the run), so callers wrap ``write_progress`` defensively.
"""

from __future__ import annotations

from pathlib import Path

from autointerp.spec import InvestigationSpec

from .run_dir import RunHandle
from .state import (
    RunState,
    StageStatus,
    Verdict,
    now_iso,
    read_state,
)

_STAGE_MARK = {
    StageStatus.COMPLETED: "x",
    StageStatus.IN_PROGRESS: ">",
    StageStatus.ABORTED: "!",
    StageStatus.PENDING: " ",
}

_VERDICT_MARK = {
    Verdict.PASS: "PASS        ",
    Verdict.FAIL: "FAIL        ",
    Verdict.INCONCLUSIVE: "INCONCLUSIVE",
}


def _load_spec(handle: RunHandle) -> InvestigationSpec | None:
    try:
        return InvestigationSpec.model_validate_json(handle.spec_path.read_text())
    except Exception:
        return None


def _stage_lines(state: RunState, spec: InvestigationSpec | None) -> list[str]:
    lines: list[str] = []
    for idx in sorted(state.stage_status, key=int):
        rec = state.stage_status[idx]
        mark = _STAGE_MARK.get(rec.status, " ")
        suffix = rec.status.value
        if rec.artifact_refs:
            suffix += f" ({len(rec.artifact_refs)} artifacts)"
        lines.append(f"- [{mark}] {idx} {rec.stage} — {suffix}")
    if not lines:
        lines.append("- (no stages)")
    return lines


def _criteria_lines(state: RunState, spec: InvestigationSpec | None) -> list[str]:
    # Iterate spec order so pending criteria are listed too; fall back to
    # whatever has been evaluated if the spec cannot be read.
    if spec is not None:
        ordered_ids = [c.criterion_id for c in spec.success_criteria]
        descriptions = {c.criterion_id: c.description for c in spec.success_criteria}
    else:
        ordered_ids = list(state.criteria_evaluated)
        descriptions = {}

    lines: list[str] = []
    for cid in ordered_ids:
        rec = state.criteria_evaluated.get(cid)
        if rec is None:
            desc = descriptions.get(cid, "")
            tail = f" — {desc}" if desc else ""
            lines.append(f"- pending      {cid}{tail}")
            continue
        mark = _VERDICT_MARK.get(rec.verdict, rec.verdict.value)
        if rec.verdict is Verdict.INCONCLUSIVE:
            body = f"observed {rec.value}; reason: {rec.inconclusive_reason}"
        else:
            body = (
                f"{rec.metric} {rec.comparator} {rec.threshold} "
                f"(observed {rec.value})"
            )
        lines.append(f"- {mark} {cid} — {body}")
    if not lines:
        lines.append("- (no success criteria)")
    return lines


def render_progress(handle: RunHandle) -> str:
    """Render the compact progress digest as markdown."""
    state = read_state(handle.state_path)
    spec = _load_spec(handle)

    n_stages = len(state.stage_status)
    idx = state.current_stage_idx
    if idx >= n_stages:
        stage_line = f"Stage: {n_stages}/{n_stages} — all stages complete"
    else:
        cur = state.stage_status.get(str(idx))
        cur_name = cur.stage if cur is not None else "?"
        stage_line = f"Stage: {idx + 1}/{n_stages} — {cur_name}"

    status = (
        state.terminal_state.value
        if state.terminal_state is not None
        else "in_progress"
    )

    b = state.budget_consumed
    budget_line = (
        f"tool_calls={b.tool_calls} bash_calls={b.bash_calls} "
        f"gpu_seconds={b.gpu_seconds} wallclock={b.wallclock_seconds}s "
        f"samples={b.samples}"
    )

    out: list[str] = [
        f"# Investigation progress — {state.run_id}",
        "",
        f"Status: {status}",
        stage_line,
        f"Updated: {now_iso()}",
        "",
        "## Stages",
        *_stage_lines(state, spec),
        "",
        "## Criteria",
        *_criteria_lines(state, spec),
        "",
        "## Budget",
        budget_line,
    ]

    if state.abort_triggered is not None:
        a = state.abort_triggered
        out += [
            "",
            "## Abort",
            f"- {a.predicate_id} on metric {a.metric} "
            f"(value={a.value}) at stage {a.at_stage_idx}",
        ]
    if state.spec_revision_requested is not None:
        out += [
            "",
            "## Revision requested",
            f"- {state.spec_revision_requested.reason}",
        ]

    return "\n".join(out) + "\n"


def write_progress(handle: RunHandle) -> Path:
    """Render and atomically replace ``progress.md`` in the run root.

    Best-effort: returns the path on success. The caller (a gate) is
    responsible for not letting a digest error abort the run.
    """
    path = handle.progress_path
    text = render_progress(handle)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return path


def refresh_progress(handle: RunHandle) -> None:
    """Best-effort ``write_progress`` for use inside gates.

    The progress digest is a convenience surface, never a correctness
    invariant. A failure here (disk full, race on the tmp file) must not
    propagate into ``advance_stage`` / ``evaluate_criterion`` and abort an
    otherwise-valid run, so all exceptions are swallowed.
    """
    try:
        write_progress(handle)
    except Exception:
        pass


__all__ = ["render_progress", "write_progress", "refresh_progress"]
