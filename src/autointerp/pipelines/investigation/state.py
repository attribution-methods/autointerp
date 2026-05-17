"""Run-level mutable bookkeeping for the investigation pipeline.

A single ``state.json`` file per run captures: which stage is active, which
artifacts have been committed, which criteria have been evaluated (write-once),
budget consumption, abort/terminal status, and any pending provenance tokens
issued by ``compute_metric``.

Writes are atomic (write-temp-then-rename) so a crash mid-write cannot leave a
partial state visible. The schema is pydantic so every field is typed and
round-trip-stable.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .flags import AblationFlags


class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABORTED = "aborted"


class TerminalState(str, Enum):
    COMPLETED = "completed"
    CRITERION_FAILED = "criterion_failed"
    ABORTED = "aborted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REVISION_REQUESTED = "revision_requested"


class Verdict(str, Enum):
    """Three-way outcome for a pre-registered criterion.

    PASS / FAIL are the threshold-comparison outcomes. INCONCLUSIVE is for
    honest non-results — the metric was computed, but a stated reason (small
    n, wide CI, data-quality issue, etc.) prevents reading a verdict off the
    value. Only FAIL flips the run to ``CRITERION_FAILED``; INCONCLUSIVE is
    terminal-state-neutral.
    """

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class StateBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StageRecord(StateBaseModel):
    stage: str
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    ended_at: str | None = None
    abort_reason: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)


class CriterionRecord(StateBaseModel):
    verdict: Verdict
    value: float
    metric: str
    comparator: str
    threshold: float
    metric_result_ref: str
    evaluated_at: str
    inconclusive_reason: str | None = None


class AbortRecord(StateBaseModel):
    predicate_id: str
    metric: str
    value: float
    at_stage_idx: int
    at: str


class BudgetConsumed(StateBaseModel):
    tool_calls: int = 0
    bash_calls: int = 0
    gpu_seconds: float = 0.0
    wallclock_seconds: float = 0.0
    samples: int = 0


class RevisionRequest(StateBaseModel):
    reason: str
    prior_results_ref: str | None = None
    requested_at: str


class ProvenanceToken(StateBaseModel):
    metric: str
    metric_id: str
    value: float
    inputs_hash: str
    issued_at: str
    stage_idx: int


class RunState(StateBaseModel):
    """Mutable bookkeeping for a single investigation run."""

    schema_version: int = 1
    spec_id: str
    spec_revision: int
    run_id: str
    run_started_at: str
    run_ended_at: str | None = None

    current_stage_idx: int = 0
    stage_status: dict[str, StageRecord] = Field(default_factory=dict)

    criteria_evaluated: dict[str, CriterionRecord] = Field(default_factory=dict)

    abort_triggered: AbortRecord | None = None
    budget_consumed: BudgetConsumed = Field(default_factory=BudgetConsumed)
    spec_revision_requested: RevisionRequest | None = None
    terminal_state: TerminalState | None = None

    pending_provenance_tokens: dict[str, ProvenanceToken] = Field(default_factory=dict)
    provenance_tokens_consumed: int = 0

    # Which discipline mechanisms are enforced for this run. Absent in
    # pre-existing state.json files → defaults to all-on (the original
    # scaffold), so old runs load and behave unchanged.
    ablation_flags: AblationFlags = Field(default_factory=AblationFlags)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def is_terminal_state(value: TerminalState | None) -> bool:
    """Single greppable predicate: the run will not transition further.

    Adapted from claude-code's ``isTerminalTaskStatus`` — a central helper
    beats scattered ``state.terminal_state in {COMPLETED, ABORTED, ...}``
    literals across gates / report / CLI.
    """
    return value is not None


def read_state(path: Path) -> RunState:
    return RunState.model_validate_json(Path(path).read_text())


def write_state(path: Path, state: RunState) -> None:
    """Atomic write: serialize to a tempfile in the same dir, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.model_dump_json(indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def initial_state(
    spec_id: str,
    spec_revision: int,
    n_stages: int,
    stage_names: list[str],
    ablation_flags: AblationFlags | None = None,
) -> RunState:
    """Build a fresh RunState with one StageRecord per spec stage, all PENDING."""
    if len(stage_names) != n_stages:
        raise ValueError(f"stage_names length {len(stage_names)} != n_stages {n_stages}")
    stage_status = {
        str(i): StageRecord(stage=stage_names[i], status=StageStatus.PENDING) for i in range(n_stages)
    }
    return RunState(
        spec_id=spec_id,
        spec_revision=spec_revision,
        run_id=f"{spec_id}_rev{spec_revision}",
        run_started_at=now_iso(),
        current_stage_idx=0,
        stage_status=stage_status,
        ablation_flags=ablation_flags or AblationFlags(),
    )


__all__ = [
    "AblationFlags",
    "RunState",
    "StageRecord",
    "StageStatus",
    "TerminalState",
    "Verdict",
    "CriterionRecord",
    "AbortRecord",
    "BudgetConsumed",
    "RevisionRequest",
    "ProvenanceToken",
    "now_iso",
    "is_terminal_state",
    "read_state",
    "write_state",
    "initial_state",
]


# Touch-test imports
_ = (json, Any)
