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
        from autointerp.spec import InvestigationSpec

        state = read_state(handle.state_path)
        spec = InvestigationSpec.model_validate_json(handle.spec_path.read_text())
        limits = {
            k: ("unlimited" if v is None else v)
            for k, v in spec.budget.model_dump().items()
        }
        return _ok({
            "limits": limits,
            "consumed_so_far": state.budget_consumed.model_dump(),
            "note": (
                "limits of 'unlimited' mean NO cap — proceed normally. "
                "consumed_so_far is what you have already used (starts at 0); "
                "it is NOT a remaining allowance. Do NOT request a spec "
                "revision over budget unless a real, non-unlimited limit here "
                "is actually exhausted."
            ),
        })

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

    async def _discover_features_handler(args: dict[str, Any]) -> tuple[str, bool]:
        from autointerp.pipelines.investigation.discovery import run_discovery_subagent
        from autointerp.spec import InvestigationSpec, ToolName

        # --- Stage gating: only legal when the active stage lists the tool. ---
        try:
            spec = InvestigationSpec.model_validate_json(handle.spec_path.read_text())
        except Exception as exc:
            return _err(exc)
        state = read_state(handle.state_path)
        idx = state.current_stage_idx
        if idx >= len(spec.stages):
            return "discover_features: no active stage to discover for", False
        stage = spec.stages[idx]
        if ToolName.DISCOVER_FEATURES not in stage.tools:
            return (
                f"discover_features: stage {stage.stage.value} (idx {idx}) does "
                f"not list `discover_features` in its tools — refused. Add it via "
                f"a spec revision, or call it only in a stage that allows it.",
                False,
            )

        task = args.get("task")
        if not isinstance(task, str) or not task:
            return "task (non-empty string) is required", False

        # --- Bind reward to the stage's pre-registered metrics. ---
        allowed = sorted({m.value for m in stage.metrics})
        reward_metric = args.get("reward_metric")
        if reward_metric is None:
            reward_metric = allowed[0] if len(allowed) == 1 else None
            if reward_metric is None:
                return (
                    f"discover_features: reward_metric is required when the stage "
                    f"lists multiple pre-registered metrics: {allowed}. Pick one.",
                    False,
                )
        elif reward_metric not in allowed:
            return (
                f"discover_features: reward_metric={reward_metric!r} is not in the "
                f"stage's pre-registered metrics: {allowed}. Pick one, or open a "
                f"spec revision via `request_spec_revision`.",
                False,
            )

        # --- Hyperparameters from the frozen DiscoveryConfig (or defaults). ---
        disco = stage.discovery
        max_iterations = int(args.get("max_iterations",
                                      disco.max_iterations if disco else 8))
        patience = disco.patience if disco else 2
        n_subagents = disco.n_subagents if disco else 1
        archive_size = disco.archive_size if disco else 5
        propose_mode = disco.propose_mode if disco else "single"
        max_turns = disco.max_turns_per_iteration if disco else 20
        top_k = disco.top_k if disco else 20
        k_grid = list(disco.k_grid) if disco else [1, 5, 10, 20, 50]
        objective = disco.objective if disco else "combined"
        seeds = list(disco.seeds) if disco else [0]
        evaluator = args.get("evaluator", disco.evaluator if disco else None)
        dry_run = bool(args.get("dry_run", False))

        # --- Model/temperature: subagent override > spec config > runtime. ---
        try:
            from autointerp_agent.config import load_config
            cfg = load_config()
            default_model, temperature = cfg.model_name, cfg.temperature
        except Exception:
            default_model, temperature = "anthropic/claude-sonnet-4-5", None
        model = str(args.get("model") or (disco.model if disco and disco.model else default_model))

        # --- Budget: cap discovery spend at the remaining token budget. ---
        max_tokens = disco.max_tokens if disco else None
        b = spec.budget
        if b.max_tokens is not None:
            remaining = max(0, b.max_tokens - state.budget_consumed.tokens)
            max_tokens = remaining if max_tokens is None else min(max_tokens, remaining)

        session_name = str(args.get("session_name") or f"stage{idx:02d}")
        session_dir = handle.discovery_dir / session_name
        try:
            result = await run_discovery_subagent(
                session_dir=session_dir,
                task=task,
                reward_metric=reward_metric,
                reward_description=(
                    f"Discovery reward `{reward_metric}` — see metrics/"
                    f"{reward_metric}.md for the contract."
                ),
                model=model,
                temperature=temperature,
                max_iterations=max_iterations,
                patience=patience,
                n_subagents=n_subagents,
                archive_size=archive_size,
                propose_mode=propose_mode,
                max_turns_per_iteration=max_turns,
                seeds=seeds,
                top_k=top_k,
                k_grid=k_grid,
                objective=objective,
                evaluator=evaluator,
                max_tokens=max_tokens,
                dry_run=dry_run,
            )
        except Exception as exc:
            return _err(exc)

        # --- Bridge discovery spend into the run budget. ---
        if result.tokens_used or result.cost_usd:
            st = read_state(handle.state_path)
            st.budget_consumed.tokens += int(result.tokens_used)
            st.budget_consumed.cost_usd = round(
                st.budget_consumed.cost_usd + float(result.cost_usd), 6
            )
            from .state import write_state
            write_state(handle.state_path, st)

        return _ok(
            {
                "session_dir": str(result.session_dir),
                "best_candidate": result.best_candidate,
                "best_reward": result.best_reward,
                "best_reward_std": result.best_reward_std,
                "best_summary": result.best_summary,
                "archive": result.archive,
                "iterations_run": result.iterations_run,
                "proposals_made": result.proposals_made,
                "terminated_by": result.terminated_by,
                "tokens_used": result.tokens_used,
                "cost_usd": result.cost_usd,
                "log": result.log,
                "next_step": (
                    "Advisory only — no gated artifact was committed. Inspect "
                    "best_summary.top_features (best_reward_std flags flukes). To "
                    "record them, build a CandidateSite / FeatureFinding and call "
                    "commit_artifact. To register the reward as a checkable value, "
                    f"call compute_metric ({reward_metric}) on a heldout split, "
                    "then commit_artifact."
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
        ToolSpec(
            name="discover_features",
            description=(
                "Spawn an iterative hill-climbing sub-agent that searches for a "
                "ranking algorithm maximizing a reward metric. Only allowed in "
                "stages whose `tools` list includes `discover_features`. The "
                "sub-agent writes algorithm_v{N}.py candidates under the run's "
                "discovery/ tree, evaluates each via the discovery harness, and "
                "returns the best candidate's reward + top features. ADVISORY: it "
                "does NOT commit gated artifacts. Afterwards, call commit_artifact "
                "(CandidateSite / FeatureFinding) and compute_metric + "
                "commit_artifact (MetricResult) to record evidence. Use "
                "dry_run=true for a deterministic smoke test (no model/LLM)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Free-text description of what to rank / discover.",
                    },
                    "reward_metric": {
                        "type": "string",
                        "description": (
                            "Reward to maximize. Must be one of the stage's "
                            "pre-registered metrics; defaults to the only one if "
                            "the stage lists a single metric."
                        ),
                    },
                    "max_iterations": {"type": "integer"},
                    "evaluator": {
                        "type": "string",
                        "description": "module:attr for the real (GPU) evaluator.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Deterministic stub iteration (no model/LLM). For CI.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Override the proposal LLM (default: runtime model).",
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Override session sub-dir name under discovery/.",
                    },
                },
                "required": ["task"],
            },
            handler=_discover_features_handler,
        ),
    ]


__all__ = ["create_investigation_tools"]
