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


def _resolve_metric_inputs(
    handle: RunHandle, raw_inputs: Any
) -> tuple[dict | None, dict, str | None]:
    """Resolve metric ``inputs`` (a dict, or a path to a JSON file) and record
    where they came from. Returns ``(inputs, provenance, error)``.

    provenance is ``{"source": "inline"}`` for a typed dict, or
    ``{"source": "file", "ref": <relpath>, "sha256": <hash>}`` for a file — so
    every MetricResult is traceable to the data it was computed from."""
    import hashlib
    from pathlib import Path

    if isinstance(raw_inputs, str):
        # A relative path may be written either under the run dir (write_file's
        # root) or under the agent's working dir (a `bash` heredoc's cwd) — try
        # both, the same way the agent's read_file/write_file resolve, so the
        # agent isn't forced to `cp` files between the two.
        if Path(raw_inputs).is_absolute():
            p = Path(raw_inputs)
        else:
            from_run = handle.root / raw_inputs
            p = from_run if from_run.exists() else (Path.cwd() / raw_inputs)
        if not p.exists():
            return None, {}, (
                f"inputs file not found: {raw_inputs} (looked under the run dir "
                "and the working dir). Write the file in your script first, then "
                "pass its path."
            )
        try:
            text = p.read_text()
            inputs = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            return None, {}, f"failed to read inputs file {p}: {exc}"
        if not isinstance(inputs, dict):
            return None, {}, f"inputs file {p} must contain a JSON object"
        sha = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Is this file a recorded, unmodified capture (input-provenance v2)?
        from autointerp.tools.provenance import capture_for_relpath, verify_capture

        cap = capture_for_relpath(handle.root, raw_inputs)
        if cap is not None and verify_capture(handle.root, cap):
            return inputs, {
                "source": "capture",
                "ref": cap["data_relpath"],
                "sha256": cap["sha256"],
                "capture_id": cap["capture_id"],
                "capture_source": cap.get("source"),
                "model_id": cap.get("model_id"),
                "n": cap.get("n"),
            }, None
        try:
            ref = p.resolve().relative_to(handle.root.resolve()).as_posix()
        except ValueError:
            ref = str(p)
        return inputs, {"source": "file", "ref": ref, "sha256": sha}, None
    if isinstance(raw_inputs, dict):
        return raw_inputs, {"source": "inline"}, None
    return None, {}, "inputs must be an object/dict or a path to a JSON file"


def _input_gate_error(handle: RunHandle, provenance: dict) -> str | None:
    """Enforce the input-provenance bar for the agent's metric tools.

    v2 (``require_captured_inputs``): inputs must be a ``model_forward`` capture
    — proven to come from a real forward pass. v1 (``require_sourced_inputs``):
    inputs must at least come from a produced file, never inline literals."""
    source = provenance.get("source")
    if getattr(handle.flags, "require_captured_inputs", False):
        if source == "capture" and provenance.get("capture_source") == "model_forward":
            return None
        return (
            "metric inputs must be a model_forward capture. In your script, run "
            "the model on the dataset, then call "
            "`autointerp.tools.provenance.record_capture(name, inputs_dict, "
            "source='model_forward', model_id=..., prompt_batch=...)` — it "
            "returns a path like 'captures/<id>.json'. Pass THAT path as "
            "`inputs`. Plain files and inline values are rejected so every "
            "verdict is proven to come from a real model run, not hand-typed "
            "numbers."
        )
    if handle.flags.require_sourced_inputs and source == "inline":
        return (
            "metric inputs must come from a file your script produced, not "
            "inline literals. Compute the inputs by running the model in a "
            "script, write them to e.g. scratch/<name>.json, then pass "
            'inputs="scratch/<name>.json". Inline values are rejected so every '
            "result is traceable to a real model run."
        )
    return None


def _blames_environment(reason: str) -> bool:
    """Does a revision reason blame the box/network/model-loading for what is
    really a fixable script error? The classic weak-driver failure: a wrong-repo
    404 (e.g. raw `from_pretrained("gpt2-small")`) or a slow load, reported as
    "hardware/network can't load a real model" and used to bail to a spec
    revision instead of fixing the script. Phrases are specific so a genuine
    methodological revision is never caught."""
    r = reason.lower()
    triggers = (
        "hardware restriction", "hardware/network", "no gpu", "without a gpu",
        "without gpu", "network restriction", "load a real model",
        "cannot load the model", "can't load the model", "could not load the model",
        "unable to load the model", "not available in the current environment",
        "offline", "no internet", "synthetic model output", "pre-warmed capture",
    )
    return any(t in r for t in triggers)


def _has_model_forward_capture(handle: RunHandle) -> bool:
    """True once at least one capture file exists — proof the model was actually
    loaded and run at least once in this run."""
    captures = handle.root / "captures"
    return captures.is_dir() and any(captures.iterdir())


def _did_any_empirical_work(handle: RunHandle) -> bool:
    """Has the agent actually run experiments in this run?

    True if it committed a metric/artifact/criterion, or wrote any file to its
    work dirs. Used to reject a `request_spec_revision` that comes before any
    real attempt — the classic weak-agent failure of treating "I haven't
    produced the metric's inputs yet" as a spec defect."""
    state = read_state(handle.state_path)
    if (
        state.provenance_tokens_consumed > 0
        or state.criteria_evaluated
        or any(rec.artifact_refs for rec in state.stage_status.values())
    ):
        return True
    for sub in ("scripts", "scratch", "findings", "activations", "generations",
                "captures"):
        d = handle.root / sub
        if d.is_dir() and any(d.iterdir()):
            return True
    return False


def _all_criteria_evaluated(handle: RunHandle) -> bool:
    """True when every pre-registered success criterion already has a verdict.

    A FAIL terminates the run on its own, so reaching here non-terminal means the
    evaluated criteria are PASS/INCONCLUSIVE — the verdict is in and there is
    nothing left to revise. Used to refuse a `request_spec_revision` that would
    otherwise strand a finished run (the agent hit an unrelated stage-gate snag
    and reached for a revision instead of just calling advance_stage to finish)."""
    from autointerp.spec import InvestigationSpec

    try:
        spec = InvestigationSpec.model_validate_json(handle.spec_path.read_text())
    except Exception:  # noqa: BLE001 — never let the guard itself crash the handler
        return False
    if not spec.success_criteria:
        return False
    state = read_state(handle.state_path)
    return all(
        c.criterion_id in state.criteria_evaluated for c in spec.success_criteria
    )


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
        inputs, provenance, err = _resolve_metric_inputs(handle, inputs)
        if err:
            return err, False
        sourcing_err = _input_gate_error(handle, provenance)
        if sourcing_err:
            return sourcing_err, False
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
                input_provenance=provenance,
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
        inputs, provenance, err = _resolve_metric_inputs(handle, inputs)
        if err:
            return err, False
        sourcing_err = _input_gate_error(handle, provenance)
        if sourcing_err:
            return sourcing_err, False
        try:
            payload, token = compute_metric(
                handle,
                metric=metric,
                metric_id=metric_id,
                inputs=inputs,
                threshold=threshold,
                comparator=comparator,
                metadata=metadata,
                input_provenance=provenance,
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
        if _all_criteria_evaluated(handle):
            return (
                "request_spec_revision rejected — every pre-registered success "
                "criterion has already been evaluated, so the run's verdict is "
                "already decided; there is nothing to revise. Do NOT request a "
                "revision over a stage-gating snag. Call advance_stage to close "
                "out the run (it terminates as COMPLETED once all criteria are "
                "evaluated, regardless of any remaining stages). If a later stage "
                "still has uncommitted work you care about, commit its metric "
                "first, then advance_stage — but the criteria verdict stands "
                "either way.",
                False,
            )
        if not _did_any_empirical_work(handle):
            return (
                "request_spec_revision rejected — you have not run a single "
                "experiment yet (no scripts run, no artifacts committed, no "
                "metric computed). A metric reporting 'missing required input "
                "keys' (e.g. logit_diff needs target_logits and foil_logits) "
                "does NOT mean the spec is wrong: those inputs are DATA YOU "
                "PRODUCE. Load the model (`from autointerp.tools.model import "
                "load_model`), run it on the dataset prompts in a script, save "
                "the logits to scratch/, then pass them to "
                "compute_and_commit_metric. Do the experiment first. "
                "request_spec_revision is only for a genuine spec contradiction "
                "you hit AFTER actually attempting the work.",
                False,
            )
        if _blames_environment(reason) and not _has_model_forward_capture(handle):
            return (
                "request_spec_revision rejected — this reads as a hardware / "
                "network / model-loading limitation, but the spec's model loads "
                "fine on this box and no model_forward capture exists yet, so the "
                "model was never successfully loaded here. The cause is almost "
                "always a SCRIPT bug: passing a display name (e.g. 'gpt2-small') "
                "to a raw `from_pretrained` 404s — use `load_model(...)`, which "
                "resolves it to the real Hub repo; or a command errored / timed "
                "out. Fix the script and re-run it. Do NOT revise the spec, switch "
                "to a smaller model, or substitute synthetic/offline outputs for "
                "real model runs — none of those is the problem.",
                False,
            )
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
                            "A PATH (string) to a JSON file your script wrote "
                            "with the metric's inputs, e.g. "
                            "'scratch/logit_diff_inputs.json'. Inline literal "
                            "values are rejected — inputs must be traceable to a "
                            "real model run."
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
                        "description": (
                            "A PATH (string) to a JSON file your script wrote "
                            "with the metric's inputs (inline literals are "
                            "rejected). For metric='custom', the file's object "
                            "must also include "
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
