"""Spec-revision exit.

When the agent decides the spec is wrong (failed criterion, hypothesis no
longer fits the evidence, methodology issue), the legitimate exit is to
record a revision request and terminate. Stage 0 then drafts a child spec
with ``parent_spec_id`` and ``prior_results_ref`` pointing at this run.
"""

from __future__ import annotations

import json

from .run_dir import RunHandle
from .state import (
    RevisionRequest,
    TerminalState,
    now_iso,
    read_state,
    write_state,
)


class RevisionGateError(Exception):
    pass


def request_spec_revision(
    handle: RunHandle,
    *,
    reason: str,
    prior_results_ref: str | None = None,
) -> RevisionRequest:
    """Record a revision request and flip the run to ``REVISION_REQUESTED``."""
    if not reason or not isinstance(reason, str):
        raise RevisionGateError("reason is required (a non-empty string)")

    state = read_state(handle.state_path)
    if state.spec_revision_requested is not None:
        return state.spec_revision_requested

    if (
        state.terminal_state is not None
        and state.terminal_state is not TerminalState.CRITERION_FAILED
    ):
        raise RevisionGateError(
            f"run is in terminal state {state.terminal_state.value!r}; "
            "request_spec_revision is only valid before terminal or after CRITERION_FAILED"
        )

    requested_at = now_iso()
    request = RevisionRequest(
        reason=reason,
        prior_results_ref=prior_results_ref,
        requested_at=requested_at,
    )
    state.spec_revision_requested = request
    state.terminal_state = TerminalState.REVISION_REQUESTED
    state.run_ended_at = requested_at
    write_state(handle.state_path, state)

    line = (
        json.dumps(
            {
                "ts": requested_at,
                "stage_idx": state.current_stage_idx,
                "tool": "request_spec_revision",
                "args": {
                    "reason": reason,
                    "prior_results_ref": prior_results_ref,
                },
                "ok": True,
                "result_summary": "revision requested; run terminal",
                "budget_after": state.budget_consumed.model_dump(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    with handle.log_path.open("a") as fh:
        fh.write(line)
    return request


__all__ = ["RevisionGateError", "request_spec_revision"]
