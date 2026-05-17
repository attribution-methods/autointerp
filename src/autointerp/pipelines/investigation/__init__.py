"""Investigation pipeline.

Executes an approved ``InvestigationSpec`` (produced by Stage 0) under a
gated tool surface. See ``docs/investigation.md`` for the full design.
"""

from .artifacts import ArtifactGateError, ArtifactRef, commit_artifact
from .criteria import CriterionGateError, evaluate_criterion
from .flags import AblationFlags
from .guards import GuardError, check_abort_predicates, enforce_budget
from .metrics import REGISTRY as METRIC_REGISTRY
from .metrics import MetricRegistryError, compute_metric
from .observer import RunObserver
from .progress import refresh_progress, render_progress, write_progress
from .report import assemble_report, write_report
from .revision import RevisionGateError, request_spec_revision
from .run_dir import RunHandle, init_run, load_run
from .stages import StageGateError, advance_stage, current_stage_view
from .state import RunState, StageStatus, TerminalState

__all__ = [
    "AblationFlags",
    "ArtifactGateError",
    "ArtifactRef",
    "CriterionGateError",
    "GuardError",
    "METRIC_REGISTRY",
    "MetricRegistryError",
    "RevisionGateError",
    "RunHandle",
    "RunObserver",
    "RunState",
    "StageGateError",
    "StageStatus",
    "TerminalState",
    "advance_stage",
    "assemble_report",
    "check_abort_predicates",
    "commit_artifact",
    "compute_metric",
    "current_stage_view",
    "enforce_budget",
    "evaluate_criterion",
    "init_run",
    "load_run",
    "refresh_progress",
    "render_progress",
    "request_spec_revision",
    "write_progress",
    "write_report",
]
