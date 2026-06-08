"""Run directory scaffolding for the investigation pipeline.

Layout (per ``docs/investigation.md``):

    runs/<spec_id>_rev<n>/
      spec.json              # frozen copy of approved spec (chmod r--)
      state.json             # mutable bookkeeping (atomic writes)
      log.jsonl              # append-only audit
      INVESTIGATION_LOG.md   # agent narrative notes
      prompt_batches/
      activations/
      generations/
      findings/stage_<idx>_<name>/
      scripts/
      scratch/
      report.json            # written at terminal state

This module owns ``init_run`` (fresh) and ``load_run`` (resume). It does not
own the gates (``commit_artifact`` / ``compute_metric`` / ``advance_stage``);
those live in sibling modules and operate on a ``RunHandle``.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from autointerp.spec import InvestigationSpec, SpecStatus

from .flags import AblationFlags
from .state import RunState, initial_state, read_state, write_state

DEFAULT_RUNS_ROOT = Path("runs")
INVESTIGATION_LOG_HEADER = "# Investigation Log\n\nAgent narrative notes for this run.\n"


@dataclass(frozen=True)
class RunHandle:
    """Resolved paths for a single run. Read-only; mutate state via gates."""

    root: Path
    spec_id: str
    spec_revision: int
    # All-on by default → byte-identical to the original scaffold.
    flags: AblationFlags = field(default_factory=AblationFlags)

    @property
    def run_id(self) -> str:
        return f"{self.spec_id}_rev{self.spec_revision}"

    @property
    def spec_path(self) -> Path:
        return self.root / "spec.json"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def log_path(self) -> Path:
        return self.root / "log.jsonl"

    @property
    def narrative_path(self) -> Path:
        return self.root / "INVESTIGATION_LOG.md"

    @property
    def prompt_batches_dir(self) -> Path:
        return self.root / "prompt_batches"

    @property
    def activations_dir(self) -> Path:
        return self.root / "activations"

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations"

    @property
    def findings_dir(self) -> Path:
        return self.root / "findings"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def scratch_dir(self) -> Path:
        return self.root / "scratch"

    @property
    def report_path(self) -> Path:
        return self.root / "report.json"

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.md"

    def stage_findings_dir(self, stage_idx: int, stage_name: str) -> Path:
        return self.findings_dir / f"stage_{stage_idx}_{stage_name}"

    def writable_roots(self) -> list[Path]:
        """Subtrees the agent's range-restricted ``write_file`` may target."""
        return [self.scripts_dir, self.scratch_dir, self.narrative_path]


def _make_subdirs(handle: RunHandle) -> None:
    for d in (
        handle.prompt_batches_dir,
        handle.activations_dir,
        handle.generations_dir,
        handle.findings_dir,
        handle.scripts_dir,
        handle.scratch_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _freeze_spec_file(spec_path: Path) -> None:
    """chmod 0o444 so the agent cannot edit the spec on disk."""
    spec_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _ensure_log_files(handle: RunHandle) -> None:
    if not handle.log_path.exists():
        handle.log_path.touch()
    if not handle.narrative_path.exists():
        handle.narrative_path.write_text(INVESTIGATION_LOG_HEADER)


def init_run(
    spec: InvestigationSpec,
    runs_root: Path | None = None,
    flags: AblationFlags | None = None,
) -> RunHandle:
    """Scaffold a fresh run directory for an approved spec.

    Refuses if the spec is not approved or if the run directory already exists
    (use ``load_run`` to resume). ``flags`` defaults to all-on (the original
    scaffold); a leave-one-out config disables exactly one discipline gate.
    """
    if spec.status != SpecStatus.APPROVED:
        raise ValueError(
            f"init_run requires an approved spec; got status={spec.status.value!r}"
        )
    flags = flags or AblationFlags()
    runs_root = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    run_id = f"{spec.spec_id}_rev{spec.revision}"
    root = runs_root / run_id
    if root.exists():
        raise FileExistsError(
            f"Run directory already exists: {root}. Use load_run to resume."
        )
    root.mkdir(parents=True, exist_ok=False)
    handle = RunHandle(
        root=root, spec_id=spec.spec_id, spec_revision=spec.revision, flags=flags
    )

    _make_subdirs(handle)

    handle.spec_path.write_text(spec.model_dump_json(indent=2) + "\n")
    # Flag A: only freeze (chmod r--) when the frozen-spec mechanism is on.
    if flags.freeze_spec:
        _freeze_spec_file(handle.spec_path)

    _ensure_log_files(handle)

    state = initial_state(
        spec_id=spec.spec_id,
        spec_revision=spec.revision,
        n_stages=len(spec.stages),
        stage_names=[s.stage.value for s in spec.stages],
        ablation_flags=flags,
    )
    write_state(handle.state_path, state)

    from .progress import refresh_progress  # deferred: progress imports run_dir

    refresh_progress(handle)
    return handle


def load_run(run_dir: Path) -> tuple[RunHandle, InvestigationSpec, RunState]:
    """Resume an existing run.

    Returns the handle, the frozen spec, and the loaded mutable state.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    spec_path = root / "spec.json"
    state_path = root / "state.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing spec.json in {root}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json in {root}")

    spec = InvestigationSpec.model_validate_json(spec_path.read_text())
    state = read_state(state_path)
    # The run's ablation config is fixed at init and persisted in state.json;
    # resume restores it (an old state.json without the field → all-on).
    handle = RunHandle(
        root=root,
        spec_id=spec.spec_id,
        spec_revision=spec.revision,
        flags=state.ablation_flags,
    )

    if state.spec_id != spec.spec_id or state.spec_revision != spec.revision:
        raise ValueError(
            "state.json identity does not match spec.json: "
            f"state=({state.spec_id}, rev{state.spec_revision}) vs "
            f"spec=({spec.spec_id}, rev{spec.revision})"
        )

    # Re-assert frozen permissions and missing log scaffolding (idempotent).
    # Flag A off → the spec is intentionally writable; do not re-freeze it.
    _make_subdirs(handle)
    if handle.flags.freeze_spec and os.access(spec_path, os.W_OK):
        _freeze_spec_file(spec_path)
    _ensure_log_files(handle)

    from .progress import refresh_progress  # deferred: progress imports run_dir

    refresh_progress(handle)

    return handle, spec, state


__all__ = ["RunHandle", "DEFAULT_RUNS_ROOT", "init_run", "load_run"]
