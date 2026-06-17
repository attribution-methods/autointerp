"""Assemble the final ``InvestigationReport`` from committed artifacts.

The report is bookkeeping: every artifact already lives on disk under the
typed schemas, so writing it is a matter of loading + grouping. Called once
at the run's terminal state.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from autointerp import schemas as S
from autointerp.spec import InvestigationSpec

from .run_dir import RunHandle
from .state import Verdict, read_state


def _load_all(dir_: Path, glob: str, cls: type[BaseModel]) -> list[Any]:
    """Load every ``glob`` match in ``dir_`` as ``cls``, skipping bad files.

    Report assembly runs at the run's terminal state and must never crash on
    stray or malformed files. The agent can write arbitrary JSON into these
    directories (e.g. a hand-rolled activation-cache sidecar in ``activations/``
    via a bash-run script), so files that aren't valid JSON, aren't readable,
    or don't match the artifact schema are skipped with a warning rather than
    aborting the whole report.
    """
    if not dir_.is_dir():
        return []
    out: list[Any] = []
    for path in sorted(dir_.glob(glob)):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            out.append(cls.model_validate(payload))
        except ValidationError as exc:
            warnings.warn(
                f"report: skipping off-schema {path.name} "
                f"(does not match {cls.__name__}): {exc.error_count()} error(s)",
                stacklevel=2,
            )
            continue
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

    # Input provenance: where each committed metric's inputs came from
    # (a produced file vs inline literals). Traceability for every result.
    metric_input_sources: dict[str, Any] = {}
    if handle.findings_dir.is_dir():
        for stage_dir in sorted(handle.findings_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            behavioral.extend(_load_all(stage_dir, "behavioral_*.json", S.BehavioralFinding))
            candidate_sites.extend(_load_all(stage_dir, "candidate_*.json", S.CandidateSite))
            feature_findings.extend(_load_all(stage_dir, "feature_*.json", S.FeatureFinding))
            validations.extend(_load_all(stage_dir, "validation_*.json", S.ValidationResult))
            for mp in sorted(stage_dir.glob("metric_*.json")):
                try:
                    mr = json.loads(mp.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                prov = (mr.get("metadata") or {}).get("input_provenance")
                if mr.get("metric_id"):
                    metric_input_sources[mr["metric_id"]] = prov or {"source": "unknown"}

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
    # Grounding check: are the evaluated criteria backed by measurements that
    # trace to a real model run?
    #   - precise (input-provenance v2): each criterion's metric must have been
    #     computed on a model_forward capture;
    #   - coarse fallback: the run produced *some* model-interaction evidence
    #     (prompt batch / activation / generation / finding / capture / script).
    def _criterion_is_model_forward(rec: Any) -> bool:
        ref = getattr(rec, "metric_result_ref", None)
        if not ref:
            return False
        mp = handle.root / ref
        if not mp.exists():
            return False
        try:
            prov = (json.loads(mp.read_text()).get("metadata") or {}).get(
                "input_provenance"
            ) or {}
        except (json.JSONDecodeError, OSError):
            return False
        return (
            prov.get("source") == "capture"
            and prov.get("capture_source") == "model_forward"
        )

    grounding_warning: str | None = None
    if state.criteria_evaluated:
        captures_present = (
            handle.captures_dir.is_dir() and any(handle.captures_dir.iterdir())
        )
        has_evidence = bool(
            prompt_batches or activation_caches or samples or behavioral
            or candidate_sites or feature_findings or validations or captures_present
        )
        scripts = (
            list(handle.scripts_dir.glob("*.py"))
            if handle.scripts_dir.is_dir() else []
        )
        coarse_ungrounded = not has_evidence and not scripts
        precise_ungrounded = False
        if getattr(handle.flags, "require_captured_inputs", False):
            backed = [
                cid for cid, rec in state.criteria_evaluated.items()
                if _criterion_is_model_forward(rec)
            ]
            precise_ungrounded = not backed  # nothing traces to a forward pass
        if coarse_ungrounded or precise_ungrounded:
            grounding_warning = (
                "RESULTS NOT GROUNDED — the evaluated criteria are not backed by "
                "measurements that trace to a real model run (no model_forward "
                "captures). Treat these verdicts as UNSUBSTANTIATED, not as "
                "findings, until the metrics are recomputed on real model output."
            )
            limitations.insert(0, grounding_warning)
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
        "grounding_warning": grounding_warning,
        "metric_input_sources": metric_input_sources,
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
