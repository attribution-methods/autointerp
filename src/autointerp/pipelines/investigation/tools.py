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
from .metrics import MetricRegistryError, compute_metric
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
        if not isinstance(inputs, dict):
            return "inputs must be an object/dict", False
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
        if not isinstance(cid, str) or not cid:
            return "criterion_id (non-empty string) is required", False
        try:
            rec = evaluate_criterion(handle, cid, metric_result_ref=ref)
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
            name="compute_metric",
            description=(
                "Run the canonical implementation of a closed-vocabulary metric "
                "and receive a one-time provenance_token. Pass the returned "
                "payload + token to commit_artifact('MetricResult', ...). The "
                "agent cannot edit the value before committing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": metric_enum},
                    "metric_id": {"type": "string"},
                    "inputs": {"type": "object", "additionalProperties": True},
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
                "Refuses on cross-split contamination. A failed criterion "
                "terminates the run; recover via request_spec_revision."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
                    "metric_result_ref": {"type": "string"},
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
