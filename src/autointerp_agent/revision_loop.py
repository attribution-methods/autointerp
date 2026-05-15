"""Auto-revision loop helpers.

When an investigation ends with ``CRITERION_FAILED`` or ``REVISION_REQUESTED``,
the next step is a child spec — a new revision pointing back at the prior
run. The unified ``investigate`` CLI re-engages Stage 0 automatically, with
the prior-run summary preloaded as the initial conversation context, so the
user doesn't have to copy-paste findings between sessions.

Pure-data helpers live here (no LLM calls) so they can be unit-tested
without an Anthropic key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Terminal states that warrant drafting a child spec automatically. Other
# terminal states (COMPLETED, ABORTED, BUDGET_EXHAUSTED) end the work.
REVISABLE_TERMINAL_STATES: frozenset[str] = frozenset(
    {"criterion_failed", "revision_requested"}
)


def should_auto_revise(state: dict[str, Any] | None) -> bool:
    """True iff the run's terminal_state warrants re-engaging Stage 0."""
    if not state:
        return False
    return str(state.get("terminal_state") or "") in REVISABLE_TERMINAL_STATES


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _format_criterion(cid: str, rec: dict[str, Any]) -> str:
    verdict = str(rec.get("verdict", "")).upper() or "?"
    if verdict == "INCONCLUSIVE":
        return (
            f"  - INCONCLUSIVE {cid!r}: {rec.get('metric')} "
            f"(observed {rec.get('value')}) — {rec.get('inconclusive_reason')}"
        )
    return (
        f"  - {verdict} {cid!r}: {rec.get('metric')} "
        f"{rec.get('comparator')} {rec.get('threshold')} "
        f"(observed {rec.get('value')})"
    )


def _tail_lines(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return lines[-n:]


def summarize_for_stage0(run_dir: Path, *, log_tail_lines: int = 60) -> str:
    """Produce a Stage-0 readable summary of a finished/half-finished run.

    Includes: terminal_state + reason, criteria evaluated, abort/revision
    record, the tail of INVESTIGATION_LOG.md (the agent's own narrative
    notes), and pointers at the on-disk findings + scripts.

    Returns Markdown — drop it straight into Stage 0 as the user's first
    message so the planner has full context for drafting a child spec.
    """
    state = _load_json(run_dir / "state.json") or {}
    report = _load_json(run_dir / "report.json") or {}
    spec = _load_json(run_dir / "spec.json") or {}

    parent_spec_id = state.get("spec_id") or spec.get("spec_id") or "<unknown>"
    parent_revision = state.get("spec_revision") or spec.get("revision") or 0
    terminal = state.get("terminal_state") or "in_progress"

    parts: list[str] = []
    parts.append(
        f"# Prior run summary: `{parent_spec_id}_rev{parent_revision}`"
    )
    parts.append(
        f"**Terminal state:** `{terminal}`  ·  "
        f"started {state.get('run_started_at','?')}, ended {state.get('run_ended_at','?')}"
    )
    if state.get("spec_revision_requested"):
        rr = state["spec_revision_requested"]
        parts.append(f"\n## Revision reason\n> {rr.get('reason','(no reason given)')}")
        if rr.get("prior_results_ref"):
            parts.append(f"\n*prior_results_ref:* `{rr['prior_results_ref']}`")
    if state.get("abort_triggered"):
        ar = state["abort_triggered"]
        parts.append(
            f"\n## Abort\n- predicate `{ar.get('predicate_id')}` on metric "
            f"`{ar.get('metric')}` (value={ar.get('value')}) at stage "
            f"{ar.get('at_stage_idx')}"
        )

    parts.append("\n## Question and hypothesis (from parent spec)")
    parts.append(f"- **question:** {spec.get('question','?')}")
    parts.append(f"- **hypothesis:** {spec.get('hypothesis','?')}")

    crits = state.get("criteria_evaluated") or {}
    if crits:
        parts.append("\n## Pre-registered criteria (evaluated)")
        for cid, rec in crits.items():
            parts.append(_format_criterion(cid, rec))
    elif spec.get("success_criteria"):
        parts.append("\n## Pre-registered criteria (none reached evaluation)")
        for crit in spec["success_criteria"]:
            parts.append(
                f"  - {crit.get('criterion_id')}: {crit.get('metric')} "
                f"{crit.get('comparator')} {crit.get('threshold')}"
            )

    findings_dir = run_dir / "findings"
    if findings_dir.is_dir():
        finding_paths = sorted(findings_dir.rglob("*.json"))
        if finding_paths:
            rels = [str(p.relative_to(run_dir)) for p in finding_paths[:10]]
            parts.append("\n## Findings on disk (first 10)")
            for r in rels:
                parts.append(f"  - `{r}`")
            if len(finding_paths) > 10:
                parts.append(f"  - … and {len(finding_paths) - 10} more")

    log_lines = _tail_lines(run_dir / "INVESTIGATION_LOG.md", n=log_tail_lines)
    if log_lines:
        parts.append("\n## INVESTIGATION_LOG.md (tail)")
        parts.append("```\n" + "\n".join(log_lines) + "\n```")

    if report.get("claims"):
        parts.append("\n## Claims (from report)")
        for c in report["claims"][:8]:
            parts.append(f"  - {c}")

    parts.append(
        "\n---\n"
        "Draft a **child spec** that addresses this. Set "
        f"`parent_spec_id={parent_spec_id!r}`, "
        f"`prior_results_ref={'runs/' + run_dir.name + '/report.json'!r}`, "
        "and a `revision_reason` that explains what changes and why "
        "(narrower hypothesis, different model, different contrast, "
        "different metric thresholds, etc.). Don't just re-run the same spec."
    )
    return "\n".join(parts)


__all__ = [
    "REVISABLE_TERMINAL_STATES",
    "should_auto_revise",
    "summarize_for_stage0",
]
