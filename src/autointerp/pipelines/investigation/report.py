"""Assemble the final ``InvestigationReport`` from committed artifacts.

The report is bookkeeping: every artifact already lives on disk under the
typed schemas, so writing it is a matter of loading + grouping. Called once
at the run's terminal state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autointerp import schemas as S
from autointerp.spec import InvestigationSpec

from .run_dir import RunHandle
from .state import Verdict, read_state


def _load_all(dir_: Path, glob: str, cls: type[BaseModel]) -> list[Any]:
    if not dir_.is_dir():
        return []
    out: list[Any] = []
    for path in sorted(dir_.glob(glob)):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out.append(cls.model_validate(payload))
    return out


def assemble_report(handle: RunHandle) -> S.InvestigationReport:
    spec = InvestigationSpec.model_validate_json(handle.spec_path.read_text())
    state = read_state(handle.state_path)

    prompt_batches = _load_all(handle.prompt_batches_dir, "*.json", S.PromptBatch)
    activation_caches = _load_all(handle.activations_dir, "*.json", S.ActivationCacheRef)
    samples = _load_all(handle.generations_dir, "*.json", S.GenerationSample)

    behavioral: list[S.BehavioralFinding] = []
    candidate_sites: list[S.CandidateSite] = []
    feature_findings: list[S.FeatureFinding] = []
    validations: list[S.ValidationResult] = []

    if handle.findings_dir.is_dir():
        for stage_dir in sorted(handle.findings_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            behavioral.extend(_load_all(stage_dir, "behavioral_*.json", S.BehavioralFinding))
            candidate_sites.extend(_load_all(stage_dir, "candidate_*.json", S.CandidateSite))
            feature_findings.extend(_load_all(stage_dir, "feature_*.json", S.FeatureFinding))
            validations.extend(_load_all(stage_dir, "validation_*.json", S.ValidationResult))

    claims: list[str] = []
    limitations: list[str] = []
    for cid, rec in state.criteria_evaluated.items():
        if rec.verdict is Verdict.INCONCLUSIVE:
            claims.append(
                f"INCONCLUSIVE criterion {cid!r}: {rec.metric} (observed "
                f"{rec.value}) — {rec.inconclusive_reason}"
            )
            limitations.append(
                f"criterion {cid!r} inconclusive: {rec.inconclusive_reason}"
            )
        else:
            claims.append(
                f"{rec.verdict.value.upper()} criterion {cid!r}: {rec.metric} "
                f"{rec.comparator} {rec.threshold} (observed {rec.value})"
            )
    if state.terminal_state is not None:
        limitations.append(f"run terminal_state={state.terminal_state.value}")
    if state.abort_triggered is not None:
        limitations.append(
            f"abort {state.abort_triggered.predicate_id} on metric "
            f"{state.abort_triggered.metric} (value={state.abort_triggered.value})"
        )

    metadata: dict[str, Any] = {
        "spec_id": spec.spec_id,
        "spec_revision": spec.revision,
        "terminal_state": (
            state.terminal_state.value if state.terminal_state is not None else None
        ),
        "criteria_evaluated": {
            cid: rec.model_dump() for cid, rec in state.criteria_evaluated.items()
        },
        "budget_consumed": state.budget_consumed.model_dump(),
    }
    cost_path = handle.root / "cost.json"
    if cost_path.exists():
        try:
            metadata["cost"] = json.loads(cost_path.read_text())
        except json.JSONDecodeError:
            pass

    report = S.InvestigationReport(
        report_id=state.run_id,
        behavior=spec.behavior,
        models=[spec.model],
        prompt_batches=prompt_batches,
        samples=samples,
        findings=behavioral,
        activation_caches=activation_caches,
        candidate_sites=candidate_sites,
        feature_findings=feature_findings,
        validations=validations,
        claims=claims,
        limitations=limitations,
        metadata=metadata,
    )
    return report


def write_report(handle: RunHandle) -> Path:
    report = assemble_report(handle)
    handle.report_path.write_text(report.model_dump_json(indent=2) + "\n")
    return handle.report_path


__all__ = ["assemble_report", "write_report"]
