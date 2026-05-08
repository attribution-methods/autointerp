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

    async def _discover_features_handler(args: dict[str, Any]) -> tuple[str, bool]:
        from autointerp.pipelines.investigation.discovery import (
            run_discovery_subagent,
        )
        from autointerp.spec import InvestigationSpec

        # Stage gating: discover_features is only legal when the active
        # stage's tools include DISCOVER_FEATURES (or the tool is unscoped
        # in spec.stages). The agent can read current_stage to confirm.
        try:
            spec = InvestigationSpec.model_validate_json(handle.spec_path.read_text())
        except Exception as exc:
            return _err(exc)
        state = read_state(handle.state_path)
        idx = state.current_stage_idx
        if idx >= len(spec.stages):
            return "discover_features: no active stage to discover for", False
        stage = spec.stages[idx]
        from autointerp.spec import ToolName
        if ToolName.DISCOVER_FEATURES not in stage.tools:
            return (
                f"discover_features: stage {stage.stage.value} (idx {idx}) "
                f"does not list `discover_features` in its tools — refused. "
                f"Either add it to the spec via revision, or call only in a "
                f"stage that allows it.",
                False,
            )

        task = args.get("task")
        if not isinstance(task, str) or not task:
            return "task (non-empty string) is required", False

        # Bind the reward metric to the spec's pre-registered list for this
        # stage. The user picks the allowed metrics at Stage 0; the master
        # agent may pick *among* them at runtime, but cannot substitute or
        # invent a new one without a spec revision.
        allowed = sorted({m.value for m in stage.metrics})
        reward_metric = args.get("reward_metric")
        if reward_metric is None:
            if len(allowed) == 1:
                reward_metric = allowed[0]
            else:
                return (
                    f"discover_features: reward_metric is required when the "
                    f"current stage lists multiple pre-registered metrics: "
                    f"{allowed}. Pick one explicitly.",
                    False,
                )
        elif reward_metric not in allowed:
            return (
                f"discover_features: reward_metric={reward_metric!r} is not "
                f"in the stage's pre-registered metrics: {allowed}. Either "
                f"pick one of those, or open a spec revision via "
                f"`request_spec_revision`.",
                False,
            )
        reward_description = args.get(
            "reward_description",
            f"Discovery reward: {reward_metric}. See metrics/{reward_metric}.md "
            "for the contract.",
        )
        max_iterations = int(args.get("max_iterations", 10))
        max_turns = int(args.get("max_turns_per_iteration", 20))
        timeout = float(args.get("iteration_timeout_s", 1200.0))
        model = str(args.get("model", "claude-sonnet-4-5"))
        permission_mode = str(args.get("permission_mode", "bypassPermissions"))
        dry_run = bool(args.get("dry_run", False))
        session_name = str(
            args.get("session_name") or f"stage{idx:02d}_call{int(state.budget_consumed.tool_calls)}"
        )

        # Pull substrate config from the spec's frozen DiscoveryConfig.
        # Validators on StageSpec already guarantee that any stage with
        # DISCOVER_FEATURES in tools also has a non-None ``discovery``
        # field, so the .substrate access here is safe.
        disco = stage.discovery
        substrate = disco.substrate if disco is not None else None
        component_kinds = (
            list(disco.component_kinds) if disco is not None else None
        )
        decomposition = disco.decomposition if disco is not None else None
        n_pairs = disco.n_pairs if disco is not None else 30
        k_grid = list(disco.k_grid) if disco is not None else None
        # The benchmark id comes from the spec's dataset.generator_id.
        benchmark_id = spec.dataset.generator_id

        session_dir = handle.discovery_dir / session_name
        try:
            result = run_discovery_subagent(
                session_dir=session_dir,
                task=task,
                reward_metric=reward_metric,
                reward_description=reward_description,
                substrate=substrate,
                component_kinds=component_kinds,
                decomposition=decomposition,
                benchmark_id=benchmark_id,
                model_id=spec.model.model_id,
                n_pairs=n_pairs,
                k_grid=k_grid,
                seed=spec.dataset.seed,
                split=spec.dataset.split,
                objective=reward_metric,
                max_iterations=max_iterations,
                max_turns_per_iteration=max_turns,
                iteration_timeout_s=timeout,
                model=model,
                permission_mode=permission_mode,
                dry_run=dry_run,
            )
        except Exception as exc:
            return _err(exc)

        return _ok(
            {
                "session_dir": str(result.session_dir),
                "best_candidate": result.best_candidate,
                "best_summary": result.best_summary,
                "iterations_run": result.iterations_run,
                "terminated_by": result.terminated_by,
                "cost_usd": result.cost_usd,
                # Surface the per-iteration log so error messages aren't
                # swallowed when terminated_by="error". The agent sees the
                # exception type + str so it can act on it (retry vs revise).
                "log": result.log,
                "next_step": (
                    "Inspect best_summary.top_features. To record them as "
                    "evidence, build a CandidateSite or FeatureFinding payload "
                    "and call commit_artifact. To register the reward as an "
                    "evaluable metric, call compute_metric with the same "
                    "reward_metric on a heldout split, then commit_artifact."
                ),
            }
        )

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
        ToolSpec(
            name="discover_features",
            description=(
                "Spawn an iterative sub-agent that hill-climbs SAE features / "
                "circuit sites against a reward metric. Only allowed in stages "
                "whose `tools` list includes `discover_features`. The sub-agent "
                "writes algorithm_v{N}.py candidates inside a session dir under "
                "the run's discovery/ tree, evaluates each via the discovery "
                "harness, and returns the best candidate's top features. The "
                "outer agent then calls commit_artifact (FeatureFinding / "
                "CandidateSite) and compute_metric+commit_artifact "
                "(MetricResult) to record evidence — discover_features itself "
                "does NOT commit gated artifacts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Free-text task description for the sub-agent.",
                    },
                    "reward_metric": {
                        "type": "string",
                        "description": "Reward to maximize (default: combined_auc_k).",
                    },
                    "reward_description": {
                        "type": "string",
                        "description": "Short paragraph summarizing the reward.",
                    },
                    "max_iterations": {"type": "integer"},
                    "max_turns_per_iteration": {"type": "integer"},
                    "iteration_timeout_s": {"type": "number"},
                    "model": {
                        "type": "string",
                        "description": "Claude model name (default claude-sonnet-4-5).",
                    },
                    "permission_mode": {
                        "type": "string",
                        "enum": ["default", "acceptEdits", "plan", "bypassPermissions"],
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "Skip the SDK and run the deterministic stub "
                            "iteration. Use for smoke tests / CI."
                        ),
                    },
                    "session_name": {
                        "type": "string",
                        "description": (
                            "Override session sub-dir name under "
                            "<run_dir>/discovery/. Defaults to a stage+call id."
                        ),
                    },
                },
                "required": ["task"],
            },
            handler=_discover_features_handler,
        ),
    ]


__all__ = ["create_investigation_tools"]
