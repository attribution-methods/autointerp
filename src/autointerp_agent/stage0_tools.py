"""Conversational Stage 0 tools.

These let the agent build an `InvestigationSpec` incrementally across turns,
retrieve phenomenon priors, validate the draft, and finalize with an Approval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from autointerp.pipelines.investigation.metrics import REGISTRY as _METRIC_REGISTRY
from autointerp.spec import (
    METRIC_META,
    CustomMetricDef,
    InvestigationSpec,
    MetricFamily,
    MetricName,
    SpecStatus,
)
from autointerp.spec_describe import describe_spec_markdown
from autointerp.spec_partial import (
    PartialSpec,
    UnknownSpecField,
    make_approval,
)

from .tools import ToolSpec


def _unregistered_metric_refs(spec: InvestigationSpec) -> list[str]:
    """Return locations referencing a MetricName with no compute fn registered.

    A metric is acceptable if (a) it is ``MetricName.CUSTOM`` (and the criterion
    supplies a ``custom_metric_def``, validated elsewhere) or (b) it appears in
    the runtime ``REGISTRY``. Anything else — typically an enum value declared
    without a registered implementation — would fail at runtime when the
    pipeline tries to evaluate it. Catch it at finalize time so the agent has
    to revise the spec before approval.
    """
    bad: list[str] = []
    registered = set(_METRIC_REGISTRY.keys()) | {MetricName.CUSTOM}
    for i, stage in enumerate(spec.stages):
        for j, m in enumerate(stage.metrics):
            if m not in registered:
                bad.append(
                    f"stages[{i}].metrics[{j}]={m.value!r} — not in metric "
                    f"registry. Use a registered metric or define a custom one."
                )
    for i, crit in enumerate(spec.success_criteria):
        if crit.metric not in registered:
            bad.append(
                f"success_criteria[{i}] (id={crit.criterion_id!r}).metric="
                f"{crit.metric.value!r} — not in metric registry. Either pick "
                f"a registered metric or set metric='custom' with a "
                f"custom_metric_def."
            )
    return bad

# Stage types that run no interventions, so they cannot produce the
# clean/corrupt/patched (etc.) inputs a causal metric needs. A causal metric
# declared here can never be committed → the run deadlocks at advance_stage.
_PRE_INTERVENTION_STAGES = {"setup", "black_box"}


def _unproducible_metric_refs(spec: InvestigationSpec) -> list[str]:
    """Return stage metrics that the stage provably cannot produce.

    Catches the most common doomed-spec shape from a weak planner: a causal
    metric (e.g. ``patch_effect_recovery``, ``faithfulness``) placed in a
    ``setup`` or ``black_box`` stage. Those stages run no interventions, so
    the metric's required inputs never exist and ``advance_stage`` can never
    succeed — better to reject at finalize than to deadlock an expensive run.
    """
    bad: list[str] = []
    for i, stage in enumerate(spec.stages):
        if stage.stage.value not in _PRE_INTERVENTION_STAGES:
            continue
        for j, m in enumerate(stage.metrics):
            meta = METRIC_META.get(m)
            if meta is not None and meta.family == MetricFamily.CAUSAL:
                bad.append(
                    f"stages[{i}] ({stage.stage.value}).metrics[{j}]={m.value!r} "
                    f"is a causal metric, but a {stage.stage.value!r} stage runs "
                    f"no interventions and cannot produce its inputs. Move it to "
                    f"an intervention/validation stage, or use a behavioral or "
                    f"localization metric here."
                )
    return bad


def _finalize_blockers(spec: InvestigationSpec) -> list[str]:
    """Every non-structural issue that ``finalize_spec`` rejects on.

    ``validate_spec`` and ``finalize_spec`` both run this so a plan that
    validates is a plan that finalizes. Without it, ``validate_spec`` reports
    "OK", the agent presents the plan and the user approves, then
    ``finalize_spec`` fails on a metric guard ``validate_spec`` never ran — and
    the agent has to silently rebuild and ask for approval a second time.
    """
    return _unregistered_metric_refs(spec) + _unproducible_metric_refs(spec)


def _render_custom_metrics_section(data: dict[str, Any]) -> str:
    """Pretty-print every CustomMetricDef referenced in success_criteria.

    The full source code is shown so the human reviewer can read it before
    approving. Once approved, source_hash is fixed and the runtime rejects
    any later change.
    """
    crits = data.get("success_criteria") or []
    if not isinstance(crits, list):
        return ""
    seen: dict[str, dict[str, Any]] = {}
    for c in crits:
        defn = c.get("custom_metric_def") if isinstance(c, dict) else None
        if isinstance(defn, dict) and defn.get("name"):
            seen.setdefault(defn["name"], defn)
    if not seen:
        return ""
    out = ["\n\n## Custom metrics — review the source code\n"]
    for name, defn in seen.items():
        out.append(f"### {name}")
        out.append(f"- description: {defn.get('description','')}")
        out.append(f"- family: {defn.get('family','?')}")
        out.append(f"- value_range: {defn.get('value_range','?')}")
        out.append(f"- direction: {defn.get('direction','?')}")
        out.append(f"- requires_inputs: {defn.get('requires_inputs','?')}")
        if defn.get("source_hash"):
            out.append(f"- source_hash: {defn['source_hash'][:16]}…")
        out.append("\n```python")
        out.append(defn.get("source_code", "<missing>"))
        out.append("```\n")
    return "\n".join(out)


PRIORS_DIR = Path(__file__).resolve().parents[2] / "priors"
METRICS_DIR = Path(__file__).resolve().parents[2] / "metrics"
SPEC_OUTPUT_DIR = Path("outputs") / "specs"
DRAFT_PATH = SPEC_OUTPUT_DIR / "_draft.json"

_partial = PartialSpec()
_pending_finalize: dict[str, Any] = {}


def _snapshot_draft() -> None:
    """Persist the in-memory partial spec to disk so it survives a restart."""
    if not _partial.data:
        return
    SPEC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_PATH.write_text(json.dumps(_partial.data, indent=2, default=str) + "\n")


def _load_draft_if_present() -> bool:
    """Restore the partial spec from disk if a draft exists. Returns whether
    anything was loaded. Idempotent — does nothing once `_partial` is populated."""
    if _partial.data:
        return False
    if not DRAFT_PATH.exists():
        return False
    try:
        _partial.data.update(json.loads(DRAFT_PATH.read_text()))
    except json.JSONDecodeError:
        return False
    return True


def _clear_draft() -> None:
    if DRAFT_PATH.exists():
        DRAFT_PATH.unlink()


async def _retrieve_prior(args: dict[str, Any]) -> tuple[str, bool]:
    phenomenon = str(args.get("phenomenon_id", "")).strip()
    if not phenomenon:
        return "phenomenon_id is required.", False
    path = PRIORS_DIR / f"{phenomenon}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PRIORS_DIR.glob("*.yaml"))
        return (
            f"No prior for {phenomenon!r}. Available: {available or '(none)'}. "
            "You may proceed without one (set phenomenon_id='custom' in update_spec).",
            False,
        )
    prior = yaml.safe_load(path.read_text()) or {}
    diff = _partial.merge_prior(prior)
    if not diff:
        return f"Prior {phenomenon!r} loaded but all fields were already set.", True
    return (
        f"Loaded prior {phenomenon!r}. Filled fields: {sorted(diff.keys())}. "
        "Existing fields were not overwritten.",
        True,
    )


async def _update_spec(args: dict[str, Any]) -> tuple[str, bool]:
    _load_draft_if_present()
    patch = args.get("patch")
    if not isinstance(patch, dict) or not patch:
        return "patch must be a non-empty object of fields to set.", False
    try:
        diff = _partial.patch(patch)
    except UnknownSpecField as exc:
        return str(exc), False
    if not diff:
        return "No changes (values matched current draft).", True
    _snapshot_draft()
    return (
        "Updated (draft auto-saved to outputs/specs/_draft.json):\n"
        + json.dumps(diff, indent=2, default=str),
        True,
    )


async def _remove_spec_fields(args: dict[str, Any]) -> tuple[str, bool]:
    _load_draft_if_present()
    keys = args.get("keys")
    if not isinstance(keys, list) or not keys:
        return "keys must be a non-empty list of top-level field names to remove.", False
    bad_types = [k for k in keys if not isinstance(k, str)]
    if bad_types:
        return f"keys must be strings; got non-strings: {bad_types}", False
    removed = _partial.remove(keys)
    if not removed:
        return f"No matching keys to remove. Current keys: {sorted(_partial.data.keys())}", True
    _snapshot_draft()
    return (
        "Removed (draft auto-saved):\n" + json.dumps(removed, indent=2, default=str),
        True,
    )


async def _show_spec(_args: dict[str, Any]) -> tuple[str, bool]:
    _load_draft_if_present()
    return (
        _partial.as_markdown() + _render_custom_metrics_section(_partial.data),
        True,
    )


async def _validate_spec(_args: dict[str, Any]) -> tuple[str, bool]:
    _load_draft_if_present()
    from pydantic import ValidationError

    missing = _partial.missing_required()
    errors: list[str] = []
    built: InvestigationSpec | None = None
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    try:
        built = _partial.try_build(status=SpecStatus.DRAFT)
    except ValidationError as exc:
        for err in exc.errors()[:25]:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            errors.append(f"{loc}: {err.get('msg', '')}")
        if exc.error_count() > 25:
            errors.append(f"... and {exc.error_count() - 25} more")
    except Exception as exc:
        errors.append(str(exc))
    # Structurally clean — now run the SAME guards finalize_spec runs, so a
    # "valid" verdict here means finalize_spec will not reject the plan after
    # the user has already approved it.
    if not errors and built is not None:
        errors.extend(_finalize_blockers(built))
    if not errors:
        return "OK — spec is valid and ready to finalize.", True
    return "Errors (must fix before finalize):\n- " + "\n- ".join(errors), True


async def _describe_spec(_args: dict[str, Any]) -> tuple[str, bool]:
    return describe_spec_markdown(), True


async def _finalize_spec(args: dict[str, Any]) -> tuple[str, bool]:
    """Lock the plan in. Single call: the agent presents the plan and gets the
    user's approval in conversation FIRST, then calls this once. It re-runs the
    full validation (so it never commits a doomed plan) and writes the approved
    spec; the investigation then launches automatically. There is no second
    "confirm" round — one approval, one finalize."""
    _load_draft_if_present()
    approver = str(args.get("approver", "")).strip()
    kind = str(args.get("approver_kind", "human")).strip()
    notes = args.get("notes")

    if not approver:
        return "approver is required (e.g. user email or agent id).", False
    if kind not in {"human", "agent"}:
        return "approver_kind must be 'human' or 'agent'.", False

    try:
        spec = _partial.try_build(status=SpecStatus.AWAITING_APPROVAL)
    except Exception as exc:
        return f"Cannot finalize — spec has structural errors:\n{exc}", False
    bad_refs = _unregistered_metric_refs(spec)
    if bad_refs:
        return (
            "Cannot finalize — the spec references metrics that are not in the "
            "runtime registry. Replace each with a RUNNABLE metric (call "
            "list_metrics to see which are runnable) or set metric='custom' "
            "with a custom_metric_def, then finalize again:\n- "
            + "\n- ".join(bad_refs),
            False,
        )
    unproducible = _unproducible_metric_refs(spec)
    if unproducible:
        return (
            "Cannot finalize — some stages declare metrics they cannot produce, "
            "which would deadlock the run at advance_stage. Fix each (via "
            "update_spec) before calling finalize_spec again:\n- "
            + "\n- ".join(unproducible),
            False,
        )

    approval = make_approval(approver=approver, kind=kind, notes=notes)
    finalized = spec.model_copy(
        update={"status": SpecStatus.APPROVED, "approval": approval}
    )
    SPEC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPEC_OUTPUT_DIR / f"{finalized.spec_id}_rev{finalized.revision}.json"
    out_path.write_text(finalized.model_dump_json(indent=2) + "\n")
    _pending_finalize.clear()
    _clear_draft()
    return (
        f"Plan finalized and saved to {out_path}. The investigation is launching "
        f"AUTOMATICALLY right now and will NOT pause for a reply. Your turn is "
        f"done: write ONE short line confirming the plan is locked and the run "
        f"is starting (e.g. \"Locked in — the investigation is running now; I'll "
        f"report back when it finishes.\"). Do NOT ask the user a question, offer "
        f"choices, or invite a response — there is no one to answer it and the "
        f"run will not wait. Do NOT run experiments, write files, or call bash. "
        f"End your turn now.",
        True,
    )


def _retrieve_prior_tool() -> ToolSpec:
    return ToolSpec(
        name="retrieve_prior",
        description=(
            "Load a phenomenon prior from priors/<id>.yaml into the draft spec. "
            "Existing fields are not overwritten. Call this first when the user "
            "names a known phenomenon (e.g. 'ioi', 'induction')."
        ),
        parameters={
            "type": "object",
            "properties": {"phenomenon_id": {"type": "string"}},
            "required": ["phenomenon_id"],
        },
        handler=_retrieve_prior,
    )


def _update_spec_tool() -> ToolSpec:
    return ToolSpec(
        name="update_spec",
        description=(
            "Patch top-level fields in the draft InvestigationSpec. "
            "Unknown keys are rejected."
        ),
        parameters={
            "type": "object",
            "properties": {"patch": {"type": "object", "additionalProperties": True}},
            "required": ["patch"],
        },
        handler=_update_spec,
    )


def _remove_spec_fields_tool() -> ToolSpec:
    return ToolSpec(
        name="remove_spec_fields",
        description="Delete top-level keys from the draft.",
        parameters={
            "type": "object",
            "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
            "required": ["keys"],
        },
        handler=_remove_spec_fields,
    )


def _show_spec_tool() -> ToolSpec:
    return ToolSpec(
        name="show_spec",
        description=(
            "Render the full InvestigationSpec shape with current values; "
            "unset fields show as _(unset)_."
        ),
        parameters={"type": "object", "properties": {}},
        handler=_show_spec,
    )


def _validate_spec_tool() -> ToolSpec:
    return ToolSpec(
        name="validate_spec",
        description=(
            "Run deterministic checks on the draft (schema, metric ranges, "
            "tool deps, split disjointness, budget)."
        ),
        parameters={"type": "object", "properties": {}},
        handler=_validate_spec,
    )


def _describe_spec_tool() -> ToolSpec:
    return ToolSpec(
        name="describe_spec",
        description=(
            "Return the InvestigationSpec schema reference: fields, types, "
            "required/optional, enum values."
        ),
        parameters={"type": "object", "properties": {}},
        handler=_describe_spec,
    )


def _finalize_spec_tool() -> ToolSpec:
    return ToolSpec(
        name="finalize_spec",
        description=(
            "Lock the plan in. Call this ONCE, after you have shown the plan "
            "and the user has approved it. It re-validates and writes the "
            "approved spec; the investigation then launches automatically. "
            "Do not ask for approval a second time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "approver": {"type": "string"},
                "approver_kind": {"type": "string", "enum": ["human", "agent"]},
                "notes": {"type": "string"},
            },
            "required": ["approver"],
        },
        handler=_finalize_spec,
    )


def create_stage0_tools(include_priors: bool = False) -> list[ToolSpec]:
    _load_draft_if_present()
    tools = [
        _describe_spec_tool(),
        _update_spec_tool(),
        _remove_spec_fields_tool(),
        _show_spec_tool(),
        _validate_spec_tool(),
        _finalize_spec_tool(),
        _list_metrics_tool(),
        _read_metric_tool(),
        _propose_custom_metric_tool(),
    ]
    if include_priors:
        tools.insert(0, _retrieve_prior_tool())
    return tools


async def _propose_custom_metric(args: dict[str, Any]) -> tuple[str, bool]:
    """Validate a candidate CustomMetricDef and echo it back for the planner
    to embed in a CUSTOM criterion via update_spec."""
    payload = dict(args)
    payload.pop("source_hash", None)  # always recomputed by the validator
    try:
        defn = CustomMetricDef.model_validate(payload)
    except Exception as exc:
        return f"Rejected — {exc}", False
    rendered = (
        f"Validated custom metric '{defn.name}' (sha256={defn.source_hash[:12]}…).\n"
        f"  family:           {defn.family.value}\n"
        f"  value_range:      {list(defn.value_range)}\n"
        f"  direction:        {defn.direction}\n"
        f"  requires_inputs:  {defn.requires_inputs}\n"
        f"\nTo use it, set `metric: \"custom\"` on a criterion in success_criteria "
        f"and put this whole block under that criterion's `custom_metric_def` "
        f"(via update_spec). Source code is frozen at finalize_spec.\n"
    )
    return rendered, True


def _propose_custom_metric_tool() -> ToolSpec:
    return ToolSpec(
        name="propose_custom_metric",
        description=(
            "Validate an inline metric definition. Use ONLY when no metric in "
            "list_metrics fits the question. The planner is not allowed to add "
            "metrics at run time; this tool only checks that a candidate "
            "definition compiles, hashes its source, and returns the validated "
            "block to embed in a Criterion's custom_metric_def field. The "
            "human reviews source_code at finalize_spec — keep the function "
            "small, deterministic, and pure-Python (math + builtins only)."
        ),
        parameters={
            "type": "object",
            "required": [
                "name", "description", "family", "value_range",
                "direction", "requires_inputs", "source_code",
            ],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "snake_case identifier; must match ^[a-z][a-z0-9_]*$",
                },
                "description": {"type": "string"},
                "family": {
                    "type": "string",
                    "enum": [f.value for f in MetricFamily],
                },
                "value_range": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                    "minItems": 2, "maxItems": 2,
                    "description": "[lo, hi]; use null for an unbounded side",
                },
                "direction": {
                    "type": "string",
                    "enum": ["higher", "lower", "either"],
                },
                "requires_inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "function_name": {
                    "type": "string",
                    "default": "compute",
                    "description": "Name of the entry function in source_code; default 'compute'",
                },
                "source_code": {
                    "type": "string",
                    "description": (
                        "Full Python source. Must define a function "
                        "`def {function_name}(inputs: dict) -> float`. "
                        "No `import` statements (the runtime pre-injects "
                        "`math`); pure-Python math/builtins only; no I/O."
                    ),
                },
            },
        },
        handler=_propose_custom_metric,
    )


async def _list_metrics(args: dict[str, Any]) -> tuple[str, bool]:
    family_filter = str(args.get("family", "")).strip().lower() or None
    if family_filter:
        try:
            wanted = MetricFamily(family_filter)
        except ValueError:
            return (
                f"Unknown family {family_filter!r}. Valid: "
                + ", ".join(f.value for f in MetricFamily),
                False,
            )
        metrics = [m for m, meta in METRIC_META.items() if meta.family == wanted]
    else:
        metrics = list(METRIC_META.keys())
    runnable: list[str] = []
    reserved: list[str] = []
    for m in metrics:
        meta = METRIC_META[m]
        lo, hi = meta.value_range
        rng = f"[{lo if lo is not None else '-∞'}, {hi if hi is not None else '∞'}]"
        if m in _METRIC_REGISTRY or m == MetricName.CUSTOM:
            runnable.append(
                f"- {m.value} ({meta.family.value}, range {rng}, "
                f"{meta.direction}-is-better): {meta.one_line}"
            )
        else:
            reserved.append(f"- {m.value} ({meta.family.value}): {meta.one_line}")
    out = [
        "RUNNABLE metrics — these have a runtime implementation; use ONLY these "
        "in stages and success_criteria:",
        *runnable,
    ]
    if reserved:
        out += [
            "",
            "NOT YET RUNNABLE — declared in the vocabulary but with no "
            "implementation. Do NOT put these in a stage or criterion (the spec "
            "will fail to finalize). If you need one of these measurements, "
            "define it inline with propose_custom_metric instead:",
            *reserved,
        ]
    return "\n".join(out), True


async def _read_metric(args: dict[str, Any]) -> tuple[str, bool]:
    name = str(args.get("name", "")).strip()
    if not name:
        return "name is required.", False
    try:
        metric = MetricName(name)
    except ValueError:
        return (
            f"Unknown metric {name!r}. Use list_metrics to see the closed vocabulary.",
            False,
        )
    meta = METRIC_META.get(metric)
    contract_block = ""
    if meta is not None:
        lo, hi = meta.value_range
        rng = f"[{lo if lo is not None else '-∞'}, {hi if hi is not None else '∞'}]"
        runnable = metric in _METRIC_REGISTRY or metric == MetricName.CUSTOM
        runnable_note = (
            "yes" if runnable
            else "NO — declared in the vocabulary but not implemented; do not "
                 "use it directly, define a custom metric instead"
        )
        contract_block = (
            f"\n\n## Contract\n"
            f"- runnable: {runnable_note}\n"
            f"- family: {meta.family.value}\n"
            f"- value_range: {rng}\n"
            f"- direction: {meta.direction}-is-better\n"
            f"- requires_inputs: {meta.requires_inputs}\n"
        )
    path = METRICS_DIR / f"{name}.md"
    body = path.read_text() if path.exists() else f"# {name}\n\n(no reference card written yet)"
    return body + contract_block, True


def _list_metrics_tool() -> ToolSpec:
    return ToolSpec(
        name="list_metrics",
        description="List metrics in the closed vocabulary, optionally filtered by family.",
        parameters={
            "type": "object",
            "properties": {"family": {"type": "string"}},
        },
        handler=_list_metrics,
    )


def _read_metric_tool() -> ToolSpec:
    return ToolSpec(
        name="read_metric",
        description="Read one metric's reference card and contract (range, direction, inputs).",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=_read_metric,
    )


def reset_stage0_state() -> None:
    """Test/utility hook to clear module-level draft state."""
    _partial.reset()
    _pending_finalize.clear()


def current_partial() -> PartialSpec:
    return _partial


__all__ = [
    "create_stage0_tools",
    "reset_stage0_state",
    "current_partial",
    "PRIORS_DIR",
    "SPEC_OUTPUT_DIR",
]


# Touch-tests imported here so the module is self-contained on import.
_ = (datetime, timezone, InvestigationSpec)
