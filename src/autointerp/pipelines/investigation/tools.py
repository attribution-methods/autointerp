"""LLM-facing wrappers for the investigation pipeline gates.

These return ``ToolSpec`` instances bound to a ``RunHandle`` so they can be
registered with the existing ``autointerp_agent.tools.ToolRouter``. Each tool
catches its module-level gate exception and returns ``(message, ok=False)``
so the agent loop can keep going.

This module deliberately does NOT import from ``autointerp_agent`` at
top-level — the import is deferred inside the factory so the pipeline package
stays usable without the agent runtime installed (e.g. for unit tests and
deterministic, scripted use).
"""

from __future__ import annotations

import json
from typing import Any

from autointerp.spec import MetricName

from .artifacts import ArtifactGateError, commit_artifact
from .criteria import CriterionGateError, evaluate_criterion
from .guards import GuardError
from .metrics import MetricRegistryError, compute_and_commit_metric, compute_metric
from .progress import render_progress
from .revision import RevisionGateError, request_spec_revision
from .run_dir import RunHandle
from .stages import StageGateError, advance_stage, current_stage_view
from .state import read_state


def _ok(value: Any) -> tuple[str, bool]:
    if isinstance(value, str):
        return value, True
    return json.dumps(value, indent=2, default=str), True


def _err(exc: Exception) -> tuple[str, bool]:
    return f"{type(exc).__name__}: {exc}", False


def create_investigation_tools(handle: RunHandle) -> list[Any]:
    """Build Tier-2 ToolSpecs bound to ``handle``.

    Returns the same ``ToolSpec`` shape used by the existing agent runtime so
    callers can register these alongside ``bash``, ``read_file`` etc.
    """
    from autointerp_agent.tools import ToolSpec  # deferred import

    async def _commit_artifact_handler(args: dict[str, Any]) -> tuple[str, bool]:
        kind = args.get("kind")
        payload = args.get("payload")
        split = args.get("split")
        token = args.get("provenance_token")
        if not isinstance(kind, str):
            return "kind must be a string", False
        if payload is None:
            return "payload is required", False
        try:
            ref = commit_artifact(
                handle, kind, payload, split=split, provenance_token=token
            )
        except ArtifactGateError as exc:
            return _err(exc)
        return _ok(
            {
                "kind": ref.kind,
                "artifact_id": ref.artifact_id,
                "relpath": ref.relpath,
                "stage_idx": ref.stage_idx,
                "split": ref.split,
            }
        )

    async def _compute_and_commit_handler(args: dict[str, Any]) -> tuple[str, bool]:
        metric = args.get("metric")
        metric_id = args.get("metric_id")
        inputs = args.get("inputs")
        split = args.get("split")
        threshold = args.get("threshold")
        comparator = args.get("comparator")
        metadata = args.get("metadata")
        criterion_id = args.get("criterion_id")
        inconclusive_reason = args.get("inconclusive_reason")
        if not isinstance(metric, str):
            return "metric (string) is required", False
        if not isinstance(metric_id, str) or not metric_id:
            return "metric_id (non-empty string) is required", False
        if isinstance(inputs, str):
            import json as _json
            from pathlib import Path as _Path
            p = _Path(inputs) if _Path(inputs).is_absolute() else handle.root / inputs
            if not p.exists():
                return f"inputs file not found: {p}", False
            try:
                inputs = _json.loads(p.read_text())
            except Exception as exc:
                return f"failed to read inputs file {p}: {exc}", False
        if not isinstance(inputs, dict):
            return "inputs must be an object/dict or a path to a JSON file", False
        if not isinstance(split, str) or not split:
            return "split (non-empty string) is required", False
        try:
            out = compute_and_commit_metric(
                handle,
                metric=metric,
                metric_id=metric_id,
                inputs=inputs,
                split=split,
                threshold=threshold,
                comparator=comparator,
                metadata=metadata,
                criterion_id=criterion_id,
                inconclusive_reason=inconclusive_reason,
            )
        except (MetricRegistryError, ArtifactGateError, CriterionGateError, GuardError) as exc:
            return _err(exc)
        return _ok(out)

    async def _compute_metric_handler(args: dict[str, Any]) -> tuple[str, bool]:
        metric = args.get("metric")
        metric_id = args.get("metric_id")
        inputs = args.get("inputs")
        threshold = args.get("threshold")
        comparator = args.get("comparator")
        metadata = args.get("metadata")
        if not isinstance(metric, str):
            return "metric (string) is required", False
        if not isinstance(metric_id, str) or not metric_id:
            return "metric_id (non-empty string) is required", False
        if isinstance(inputs, str):
            import json as _json
            from pathlib import Path as _Path
            p = _Path(inputs) if _Path(inputs).is_absolute() else handle.root / inputs
            if not p.exists():
                return f"inputs file not found: {p}", False
            try:
                inputs = _json.loads(p.read_text())
            except Exception as exc:
                return f"failed to read inputs file {p}: {exc}", False
        if not isinstance(inputs, dict):
            return "inputs must be an object/dict or a path to a JSON file", False
        try:
            payload, token = compute_metric(
                handle,
                metric=metric,
                metric_id=metric_id,
                inputs=inputs,
                threshold=threshold,
                comparator=comparator,
                metadata=metadata,
            )
        except (MetricRegistryError, GuardError) as exc:
            return _err(exc)
        return _ok({"payload": payload, "provenance_token": token})

    async def _evaluate_criterion_handler(args: dict[str, Any]) -> tuple[str, bool]:
        cid = args.get("criterion_id")
        ref = args.get("metric_result_ref")
        inconclusive_reason = args.get("inconclusive_reason")
        if not isinstance(cid, str) or not cid:
            return "criterion_id (non-empty string) is required", False
        try:
            rec = evaluate_criterion(
                handle,
                cid,
                metric_result_ref=ref,
                inconclusive_reason=inconclusive_reason,
            )
        except CriterionGateError as exc:
            return _err(exc)
        return _ok({"criterion_id": cid, **rec.model_dump()})

    async def _advance_stage_handler(_args: dict[str, Any]) -> tuple[str, bool]:
        try:
            out = advance_stage(handle)
        except (StageGateError, GuardError) as exc:
            return _err(exc)
        return _ok(out)

    async def _current_stage_handler(_args: dict[str, Any]) -> tuple[str, bool]:
        return _ok(current_stage_view(handle))

    async def _get_state_handler(_args: dict[str, Any]) -> tuple[str, bool]:
        state = read_state(handle.state_path)
        return _ok(state.model_dump())

    async def _get_budget_handler(_args: dict[str, Any]) -> tuple[str, bool]:
        state = read_state(handle.state_path)
        return _ok(state.budget_consumed.model_dump())

    async def _get_progress_handler(_args: dict[str, Any]) -> tuple[str, bool]:
        return render_progress(handle), True

    async def _request_revision_handler(args: dict[str, Any]) -> tuple[str, bool]:
        reason = args.get("reason")
        prior = args.get("prior_results_ref")
        if not isinstance(reason, str) or not reason:
            return "reason (non-empty string) is required", False
        try:
            req = request_spec_revision(handle, reason=reason, prior_results_ref=prior)
        except RevisionGateError as exc:
            return _err(exc)
        return _ok(req.model_dump())

    metric_enum = sorted(m.value for m in MetricName)

    return [
        ToolSpec(
            name="commit_artifact",
            description=(
                "Validate and persist a typed artifact under the run directory. "
                "Required: kind (e.g. 'PromptBatch', 'BehavioralFinding', "
                "'CandidateSite', 'InterventionResult', 'ValidationResult', "
                "'ActivationCacheRef', 'GenerationSample', 'MetricResult', "
                "'FeatureFinding') and payload (the typed schema as a JSON object). "
                "Required for all kinds except PromptBatch: split. Required for "
                "MetricResult only: provenance_token from compute_metric."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "payload": {"type": "object", "additionalProperties": True},
                    "split": {"type": "string"},
                    "provenance_token": {"type": "string"},
                },
                "required": ["kind", "payload"],
            },
            handler=_commit_artifact_handler,
        ),
        ToolSpec(
            name="compute_and_commit_metric",
            description=(
                "PREFERRED: one-shot metric flow. Computes the canonical "
                "metric, commits the MetricResult (using the issued provenance "
                "token), and — if `criterion_id` is given — evaluates that "
                "success criterion against this result. Replaces the manual "
                "compute_metric → commit_artifact → evaluate_criterion dance.\n"
                "When metric='custom', include '__custom_name__' in `inputs` "
                "naming the custom_metric_def in the spec.\n"
                "Set `inconclusive_reason` (with `criterion_id`) to record the "
                "criterion as INCONCLUSIVE instead of PASS/FAIL — for honest "
                "non-results (too few samples, wide CI, degenerate data). "
                "INCONCLUSIVE does not terminate the run."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": metric_enum},
                    "metric_id": {"type": "string"},
                    "inputs": {
                        "description": (
                            "Metric inputs as a dict, OR a path (string) to a "
                            "JSON file containing the inputs dict. Use a file "
                            "path when inputs are large (e.g. per-sample arrays)."
                        ),
                    },
                    "split": {
                        "type": "string",
                        "description": "Split tag for the MetricResult (e.g. 'dev', 'heldout').",
                    },
                    "threshold": {"type": "number"},
                    "comparator": {
                        "type": "string",
                        "enum": [">=", ">", "<=", "<", "=="],
                    },
                    "metadata": {"type": "object", "additionalProperties": True},
                    "criterion_id": {
                        "type": "string",
                        "description": "Optional pre-registered criterion to evaluate.",
                    },
                    "inconclusive_reason": {
                        "type": "string",
                        "description": (
                            "If set (with criterion_id), record the criterion "
                            "INCONCLUSIVE with this reason instead of PASS/FAIL."
                        ),
                    },
                },
                "required": ["metric", "metric_id", "inputs", "split"],
            },
            handler=_compute_and_commit_handler,
        ),
        ToolSpec(
            name="compute_metric",
            description=(
                "Run the canonical implementation of a closed-vocabulary metric "
                "and receive a one-time provenance_token. Pass the returned "
                "payload + token to commit_artifact('MetricResult', ...). The "
                "agent cannot edit the value before committing.\n"
                "When metric='custom', the spec may define multiple custom "
                "metrics; you must disambiguate by including a "
                "'__custom_name__' key in `inputs` whose value matches the "
                "`name` field of the relevant `custom_metric_def` in the "
                "spec. The runtime then uses that def's source_code to compute "
                "the value. Required-input keys (besides '__custom_name__') "
                "come from the def's `requires_inputs`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": metric_enum},
                    "metric_id": {"type": "string"},
                    "inputs": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": (
                            "Required-input map for the metric. For "
                            "metric='custom', also include "
                            "'__custom_name__': '<custom_metric_def.name>'."
                        ),
                    },
                    "threshold": {"type": "number"},
                    "comparator": {
                        "type": "string",
                        "enum": [">=", ">", "<=", "<", "=="],
                    },
                    "metadata": {"type": "object", "additionalProperties": True},
                },
                "required": ["metric", "metric_id", "inputs"],
            },
            handler=_compute_metric_handler,
        ),
        ToolSpec(
            name="evaluate_criterion",
            description=(
                "One-shot evaluation of a pre-registered success criterion. "
                "Refuses on cross-split contamination. A FAILed criterion "
                "terminates the run; recover via request_spec_revision. Pass "
                "`inconclusive_reason` to record INCONCLUSIVE instead — the "
                "metric value is still recorded but no PASS/FAIL is read off "
                "it and the run continues. Use for honest non-results, not to "
                "dodge a fail you can defend."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
                    "metric_result_ref": {"type": "string"},
                    "inconclusive_reason": {
                        "type": "string",
                        "description": (
                            "Why the metric cannot yield a PASS/FAIL verdict "
                            "(e.g. n too small, CI too wide, degenerate data)."
                        ),
                    },
                },
                "required": ["criterion_id"],
            },
            handler=_evaluate_criterion_handler,
        ),
        ToolSpec(
            name="advance_stage",
            description=(
                "Close the current spec stage and step to the next. Refuses if "
                "the stage has not committed a MetricResult for every declared "
                "metric, or if any AbortPredicate has tripped. Final advance "
                "requires all success criteria evaluated."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_advance_stage_handler,
        ),
        ToolSpec(
            name="current_stage",
            description=(
                "Read-only view of the active spec stage: index, name, allowed "
                "tools, declared metrics, notes, and which metrics have been "
                "committed so far this stage."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_current_stage_handler,
        ),
        ToolSpec(
            name="get_state",
            description="Read-only dump of the run's state.json (budget, stages, criteria).",
            parameters={"type": "object", "properties": {}},
            handler=_get_state_handler,
        ),
        ToolSpec(
            name="get_budget",
            description="Read-only view of consumed budget (tool_calls, gpu_seconds, samples, …).",
            parameters={"type": "object", "properties": {}},
            handler=_get_budget_handler,
        ),
        ToolSpec(
            name="get_progress",
            description=(
                "Compact progress digest: stage checklist, per-criterion "
                "verdicts (PASS/FAIL/INCONCLUSIVE or pending), budget, and any "
                "abort. Cheap to call — use it to re-ground after a long "
                "stretch of bash work instead of re-reading state.json or the "
                "transcript. Same content as the run's progress.md."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_get_progress_handler,
        ),
        ToolSpec(
            name="request_spec_revision",
            description=(
                "Terminal exit when the spec is wrong. Records a revision "
                "request and flips the run to REVISION_REQUESTED. Stage 0 then "
                "drafts a child spec with parent_spec_id pointing at this run."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "prior_results_ref": {"type": "string"},
                },
                "required": ["reason"],
            },
            handler=_request_revision_handler,
        ),
    ]


__all__ = ["create_investigation_tools"]
