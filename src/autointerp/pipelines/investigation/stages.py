"""Stage transitions and the read-only current-stage view.

``advance_stage`` is the agent's one-way door from spec stage N to N+1. It
verifies that the current stage produced a ``MetricResult`` for every metric
declared in ``stage.metrics`` and that no abort predicate has tripped, then
flips the stage to ``COMPLETED``. When the last stage is advanced past, the
gate verifies that every ``success_criterion`` has been evaluated; if any
remain unevaluated, advancement is refused. Otherwise the run terminates as
``COMPLETED`` (or stays at ``CRITERION_FAILED`` if a criterion failed earlier).

``current_stage_view`` is the read-only snapshot the agent uses to plan its
work in the active stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autointerp.spec import InvestigationSpec, MetricName

from .guards import check_abort_predicates, enforce_budget
from .run_dir import RunHandle
from .state import (
    StageStatus,
    TerminalState,
    now_iso,
    read_state,
    write_state,
)

# Maps spec ToolName values to skill directory names under skills/. Some tools
# share a skill (e.g. path_patching reuses activation-patching's workflow); a
# missing entry falls back to the tool name with underscores → hyphens.
_TOOL_TO_SKILL: dict[str, str] = {
    "blackbox_probe": "black-box-auditing",
    "linear_probe": "linear-probes",
    "logit_lens": "logit-lens",
    "direct_logit_attribution": "logit-lens",
    "activation_patching": "activation-patching",
    "path_patching": "circuit-tracing",
    "attribution_patching": "attribution-patching",
    "sae_inspect": "sparse-autoencoders",
    "steering": "activation-steering",
    "contrastive_directions": "contrastive-directions",
}


def _skill_body(tool_value: str, skills_root: Path) -> str | None:
    """Return the SKILL.md body for a tool, or None if no matching file."""
    name = _TOOL_TO_SKILL.get(tool_value, tool_value.replace("_", "-"))
    path = skills_root / name / "SKILL.md"
    if not path.is_file():
        return None
    try:
        return path.read_text()
    except OSError:
        return None


class StageGateError(Exception):
    pass


def _load_spec(handle: RunHandle) -> InvestigationSpec:
    return InvestigationSpec.model_validate_json(handle.spec_path.read_text())


def _committed_metric_names(handle: RunHandle, stage_idx: int) -> set[str]:
    """Set of MetricName values for which at least one MetricResult exists in this stage."""
    out: set[str] = set()
    state = read_state(handle.state_path)
    rec = state.stage_status.get(str(stage_idx))
    if rec is None:
        return out
    stage_dir = handle.stage_findings_dir(stage_idx, rec.stage)
    if not stage_dir.is_dir():
        return out
    for path in stage_dir.glob("metric_*.json"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        md = payload.get("metadata") or {}
        name = md.get("metric_name")
        if isinstance(name, str):
            out.add(name)
    return out


def current_stage_view(
    handle: RunHandle,
    *,
    skills_root: Path | None = None,
) -> dict[str, Any]:
    """Read-only snapshot of the active stage for the agent.

    When ``skills_root`` is provided (or when the default ``./skills`` exists
    relative to the current working directory), the view includes a
    ``skills`` field: one full SKILL.md body per tool listed in
    ``stage.tools``. This closes the discovery gap where the agent didn't
    know to call ``read_skill``.
    """
    state = read_state(handle.state_path)
    spec = _load_spec(handle)
    idx = state.current_stage_idx
    if idx >= len(spec.stages):
        return {
            "stage_idx": idx,
            "n_stages": len(spec.stages),
            "complete": True,
            "terminal_state": (
                state.terminal_state.value if state.terminal_state is not None else None
            ),
        }
    stage = spec.stages[idx]
    rec = state.stage_status[str(idx)]
    view: dict[str, Any] = {
        "stage_idx": idx,
        "n_stages": len(spec.stages),
        "stage": stage.stage.value,
        "pattern": stage.pattern.value,
        "tools": [t.value for t in stage.tools],
        "metrics": [m.value for m in stage.metrics],
        "notes": stage.notes,
        "status": rec.status.value,
        "artifact_refs": list(rec.artifact_refs),
        "committed_metric_names": sorted(_committed_metric_names(handle, idx)),
        "complete": False,
        "terminal_state": (
            state.terminal_state.value if state.terminal_state is not None else None
        ),
    }
    root = skills_root if skills_root is not None else Path.cwd() / "skills"
    if root.is_dir():
        skills: dict[str, str] = {}
        for tool in stage.tools:
            body = _skill_body(tool.value, root)
            if body is not None:
                skills[tool.value] = body
        if skills:
            view["skills"] = skills
    return view


def advance_stage(handle: RunHandle) -> dict[str, Any]:
    """Close the current stage and step to the next one (or terminate)."""
    enforce_budget(handle)
    state = read_state(handle.state_path)
    if state.terminal_state is not None:
        raise StageGateError(
            f"run is terminal ({state.terminal_state.value}); cannot advance"
        )

    spec = _load_spec(handle)
    idx = state.current_stage_idx
    if idx >= len(spec.stages):
        raise StageGateError(
            f"current_stage_idx={idx} is past the last stage ({len(spec.stages)}); "
            "nothing to advance"
        )

    stage = spec.stages[idx]
    declared = {m.value for m in stage.metrics}
    committed = _committed_metric_names(handle, idx)
    missing = sorted(declared - committed)
    if missing:
        raise StageGateError(
            f"stage {idx} ({stage.stage.value}) has not committed a MetricResult "
            f"for declared metrics: {missing}. Compute and commit them before "
            f"advancing. If a declared metric genuinely cannot be produced in "
            f"this stage (e.g. a causal metric like patch_effect_recovery needs "
            f"clean/corrupt/patched intervention values that this stage does not "
            f"create), do NOT keep retrying — call request_spec_revision to fix "
            f"the plan."
        )

    # One last guard sweep before we close the stage.
    abort = check_abort_predicates(handle)
    if abort is not None:
        raise StageGateError(
            f"abort predicate {abort.predicate_id!r} tripped on metric "
            f"{abort.metric!r} (value={abort.value}); run is terminal"
        )

    state = read_state(handle.state_path)  # re-read in case guards mutated
    rec = state.stage_status[str(idx)]
    rec.status = StageStatus.COMPLETED
    rec.ended_at = now_iso()
    state.current_stage_idx = idx + 1

    if state.current_stage_idx >= len(spec.stages):
        # Final advancement: check all criteria evaluated.
        unevaluated = [
            c.criterion_id
            for c in spec.success_criteria
            if c.criterion_id not in state.criteria_evaluated
        ]
        if unevaluated:
            # Roll back the index so the agent can call evaluate_criterion.
            state.current_stage_idx = idx
            rec.status = StageStatus.IN_PROGRESS
            rec.ended_at = None
            write_state(handle.state_path, state)
            raise StageGateError(
                f"cannot terminate: {len(unevaluated)} criteria unevaluated: "
                f"{unevaluated}"
            )
        if state.terminal_state is None:
            state.terminal_state = TerminalState.COMPLETED
            state.run_ended_at = rec.ended_at

    write_state(handle.state_path, state)

    line = (
        json.dumps(
            {
                "ts": rec.ended_at,
                "stage_idx": idx,
                "tool": "advance_stage",
                "args": {},
                "ok": True,
                "result_summary": (
                    f"stage {idx} ({stage.stage.value}) -> COMPLETED; "
                    f"new current_stage_idx={state.current_stage_idx}; "
                    f"terminal_state="
                    f"{state.terminal_state.value if state.terminal_state else None}"
                ),
                "budget_after": state.budget_consumed.model_dump(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    with handle.log_path.open("a") as fh:
        fh.write(line)

    from .progress import refresh_progress

    refresh_progress(handle)

    return {
        "advanced_from": idx,
        "current_stage_idx": state.current_stage_idx,
        "terminal_state": (
            state.terminal_state.value if state.terminal_state is not None else None
        ),
    }


__all__ = ["StageGateError", "advance_stage", "current_stage_view"]


_ = MetricName
