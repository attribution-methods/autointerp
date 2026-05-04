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
    passed: bool
    value: float
    metric: str
    comparator: str
    threshold: float
    metric_result_ref: str
    evaluated_at: str


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


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


def initial_state(spec_id: str, spec_revision: int, n_stages: int, stage_names: list[str]) -> RunState:
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
    )


__all__ = [
    "RunState",
    "StageRecord",
    "StageStatus",
    "TerminalState",
    "CriterionRecord",
    "AbortRecord",
    "BudgetConsumed",
    "RevisionRequest",
    "ProvenanceToken",
    "now_iso",
    "read_state",
    "write_state",
    "initial_state",
]


# Touch-test imports
_ = (json, Any)
