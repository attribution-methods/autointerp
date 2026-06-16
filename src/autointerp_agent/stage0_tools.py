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
from autointerp.schemas import BehaviorSpec
from autointerp.spec import (
    METRIC_META,
    Budget,
    ContrastSpec,
    Criterion,
    CustomMetricDef,
    DatasetSpec,
    InvestigationSpec,
    MetricFamily,
    MetricName,
    ModelRef,
    SpecStatus,
    StageSpec,
)
from autointerp.spec_describe import describe_spec_markdown
from autointerp.spec_partial import (
    ALLOWED_TOP_LEVEL,
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
    return (
        _unregistered_metric_refs(spec)
        + _unproducible_metric_refs(spec)
        + _custom_metric_runtime_blockers(spec)
        + _custom_without_def_blockers(spec)
        + _placeholder_model_blockers(spec)
    )


def _placeholder_model_blockers(spec: InvestigationSpec) -> list[str]:
    """The model field must be a real, loadable Hugging Face repo — not a template
    like ``local:/path/to/pythia-125M``, ``<model>``, or ``your-model``. A
    placeholder passes ``resolve_model_id`` untouched and only fails when the
    investigation tries to load weights, which a weak driver misreads as "can't
    load any model" and turns into a confusing mid-run plan-revision bail. Catch
    it at plan time so the agent picks a concrete repo before approval."""
    from autointerp.tools.model_aliases import looks_like_placeholder_model

    model_id = (spec.model.model_id or "").strip()
    if looks_like_placeholder_model(model_id):
        shown = model_id or "(empty)"
        return [
            f"the model id {shown!r} is a placeholder, not a loadable Hugging Face "
            "repo — the run would fail to load weights. Set a concrete open-weights "
            "repo id: use 'gpt2' (124M, always available) for a small/medium local "
            "model, or 'gpt2-medium' / 'EleutherAI/pythia-160m'. Never a path, an "
            "angle-bracket template, or 'local:/...'."
        ]
    return []


def _custom_without_def_blockers(spec: InvestigationSpec) -> list[str]:
    """A STAGE (not just a criterion) that lists 'custom' needs a custom_metric_def
    in the spec — otherwise the runtime has no definition to compute and the run
    stalls mid-investigation. The criterion validator only covers custom
    *criteria*, so a stray 'custom' in a stage's metrics slips through to a bail."""
    refs_custom = any(MetricName.CUSTOM in st.metrics for st in spec.stages) or any(
        c.metric == MetricName.CUSTOM for c in spec.success_criteria
    )
    has_def = any(c.custom_metric_def is not None for c in spec.success_criteria)
    if refs_custom and not has_def:
        return [
            "a stage or criterion uses metric 'custom' but the spec defines no "
            "custom_metric_def. Either define it (propose_custom_metric, then put "
            "the def on the criterion), or replace 'custom' with a RUNNABLE "
            "built-in metric (e.g. logit_diff, accuracy, kl_to_clean, ablation_drop)."
        ]
    return []


def _custom_metric_runtime_blockers(spec: InvestigationSpec) -> list[str]:
    """Dry-run every custom metric so a runtime bug (an undefined name like
    `sqrt`, a bad import) is caught at validate/finalize — not after a model run,
    where it forces a needless spec revision."""
    from autointerp.pipelines.investigation.metrics import preflight_custom_metric

    seen: set[str] = set()
    bad: list[str] = []
    for crit in spec.success_criteria:
        defn = crit.custom_metric_def
        if defn is None or defn.name in seen:
            continue
        seen.add(defn.name)
        err = preflight_custom_metric(defn)
        if err:
            bad.append(err)
    return bad


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

# Approval gate: the interactive REPL turns this on so a plan can NEVER be
# finalized (and auto-launched) on a turn where the user has not given a
# go-ahead — the weak driver otherwise drafts AND finalizes on turn 0, running
# a plan the user never saw. Off by default so non-interactive callers/tests
# (which finalize directly) are unaffected.
_approval_required = False
_user_approved = False


def set_approval_gate(required: bool) -> None:
    """REPL hook: require an explicit user go-ahead before finalize_spec writes."""
    global _approval_required
    _approval_required = required


def set_user_approved(approved: bool) -> None:
    """REPL hook: record whether THIS turn's user message was an approval."""
    global _user_approved
    _user_approved = approved


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


# Typed sub-objects the agent fills in. Their SHAPE is validated the instant
# the agent sets them (not only at finalize), so a sentence-for-behavior or a
# {source, model_id} model is caught while building — with the correct shape
# shown — instead of blowing up at the approval gate, where the weak driver
# tends to paste the schema rule to the user and ask THEM to supply field shapes.
_TYPED_OBJECT_FIELDS = {
    "behavior": BehaviorSpec,
    "model": ModelRef,
    "dataset": DatasetSpec,
    "contrast": ContrastSpec,
    "budget": Budget,
}
_TYPED_LIST_FIELDS = {
    "success_criteria": Criterion,
    "stages": StageSpec,
}

# Concrete, VALID examples (with real enum values) so a weak model can copy the
# shape instead of guessing — the repeated `stages[5]` failure was the agent not
# knowing the StageSpec shape or the valid stage/pattern enum values.
_FIELD_EXAMPLES = {
    "behavior": '{"behavior_id": "induction_copying", '
                '"description": "<one sentence describing the behavior>"}',
    "model": '{"model_id": "gpt2"}',
    "dataset": '{"dataset_id": "contrastive_prompts", "source": "generated", '
               '"n_samples": 400}',
    "contrast": '{"contrast_id": "c1", "positive_template": "...", '
                '"negative_template": "..."}',
    "stages": '{"stage": "black_box", "pattern": "blackbox_then_patching", '
              '"tools": ["blackbox_probe"], "metrics": ["accuracy"]}',
    "success_criteria": '{"criterion_id": "c1", "description": "dev accuracy", '
                        '"metric": "accuracy", "comparator": ">=", '
                        '"threshold": 0.7, "on_split": "dev"}',
}


def _explain_validation_error(exc: Any) -> str:
    """Turn a pydantic ValidationError into the SPECIFIC reasons, so the agent
    fixes the ACTUAL problem. The old generic 'wrong shape — required keys …'
    misdirected it: a semantic error ('comparator <= is wrong for the
    higher-is-better metric logit_diff') read as a missing-key problem, so the
    agent re-sent the same keys forever."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "__root__")
        where = f"'{loc}'" if loc else "value"
        etype = err.get("type", "")
        msg = err.get("msg", "invalid")
        if etype == "missing":
            parts.append(f"missing required field {where}")
        elif etype == "extra_forbidden":
            parts.append(f"unknown field {where} (remove it)")
        elif etype.startswith("value_error"):
            parts.append(msg.replace("Value error, ", ""))  # the validator's text
        else:
            parts.append(f"{where}: {msg}")
    # de-dup, keep order
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return "; ".join(uniq[:4]) or "invalid"


def _shape_error(label: str, value: Any, cls: type, exc: Any = None) -> str:
    base = label.split("[")[0]  # "stages[5]" -> "stages"
    example = _FIELD_EXAMPLES.get(base)
    detail = _explain_validation_error(exc) if exc is not None else "wrong shape"
    got = json.dumps(value, default=str)
    got = got if len(got) <= 160 else got[:160] + "…"
    msg = f"`{label}` is invalid: {detail}."
    if example:
        msg += f" A valid one looks like: {example}."
    return (
        msg + f" You passed: {got}. Fix it and call update_spec again — fix it "
        "SILENTLY in your draft; do NOT show this error to the user or ask them "
        "for field shapes; that is your job."
    )


def _validate_field_shapes(patch: dict[str, Any]) -> str | None:
    """Return an actionable error if a typed field in ``patch`` is invalid —
    naming the SPECIFIC reason (missing key, unknown key, or a semantic rule like
    a comparator/direction mismatch), not a generic 'wrong shape'."""
    from pydantic import ValidationError

    for field_name, cls in _TYPED_OBJECT_FIELDS.items():
        value = patch.get(field_name)
        if value is None:
            continue
        try:
            cls.model_validate(value)
        except ValidationError as exc:
            return _shape_error(field_name, value, cls, exc)
    for field_name, cls in _TYPED_LIST_FIELDS.items():
        value = patch.get(field_name)
        if not isinstance(value, list):
            continue
        for i, item in enumerate(value):
            try:
                cls.model_validate(item)
            except ValidationError as exc:
                return _shape_error(f"{field_name}[{i}]", item, cls, exc)
    return None


def _coerce_field_shapes(patch: dict[str, Any]) -> dict[str, Any]:
    """Best-effort fix for the shapes weak models routinely botch, so they JUST
    WORK instead of erroring. Only unambiguous, safe corrections:
      * a bare string for behavior/model becomes the right object;
      * a typed object/list-item with stray keys is trimmed to its real fields
        ONLY when that trim makes it valid (so {'source','model_id'} -> drop the
        spurious 'source'). Genuinely-unfixable shapes pass through unchanged and
        hit _validate_field_shapes' actionable error."""
    from pydantic import ValidationError

    out = dict(patch)
    if isinstance(out.get("behavior"), str) and out["behavior"].strip():
        out["behavior"] = {
            "behavior_id": "target_behavior",
            "description": out["behavior"].strip(),
        }
    if isinstance(out.get("model"), str) and out["model"].strip():
        out["model"] = {"model_id": out["model"].strip()}

    def _trim(value: Any, cls: type) -> Any:
        if not isinstance(value, dict):
            return value
        trimmed = {k: v for k, v in value.items() if k in cls.model_fields}
        if trimmed == value:
            return value
        try:
            cls.model_validate(trimmed)
            return trimmed  # the only problem was stray keys → accept the trim
        except ValidationError:
            return value    # still invalid → leave it for the actionable error

    for field_name, cls in _TYPED_OBJECT_FIELDS.items():
        if field_name in out:
            out[field_name] = _trim(out[field_name], cls)
    for field_name, cls in _TYPED_LIST_FIELDS.items():
        if isinstance(out.get(field_name), list):
            out[field_name] = [_trim(item, cls) for item in out[field_name]]
    return out


def _validate_metric_refs(patch: dict[str, Any]) -> str | None:
    """Catch metrics that are valid enum values but have NO implementation (e.g.
    mutual_information, causal_indirect_effect) the moment the agent puts them in
    a stage or criterion — not at finalize, where 'metrics not in the runtime
    registry' surfaces AFTER the user approved and the run spaghettifies."""
    runnable = {m.value for m in _METRIC_REGISTRY} | {"custom"}
    bad: set[str] = set()
    for crit in patch.get("success_criteria") or []:
        if isinstance(crit, dict):
            mv = getattr(crit.get("metric"), "value", crit.get("metric"))
            if isinstance(mv, str) and mv not in runnable:
                bad.add(mv)
    for stage in patch.get("stages") or []:
        if isinstance(stage, dict):
            for m in stage.get("metrics") or []:
                mv = getattr(m, "value", m)
                if isinstance(mv, str) and mv not in runnable:
                    bad.add(mv)
    if not bad:
        return None
    usable = ", ".join(sorted(runnable - {"custom"}))
    return (
        f"These metrics have no implementation and cannot run: {sorted(bad)}. "
        f"Use a RUNNABLE metric ({usable}), or set metric='custom' and define it "
        "with propose_custom_metric. (list_metrics shows which are runnable.) Fix "
        "it silently in your draft; do not surface this to the user."
    )


async def _update_spec(args: dict[str, Any]) -> tuple[str, bool]:
    _load_draft_if_present()
    patch = args.get("patch")
    if not isinstance(patch, dict) or not patch:
        return "patch must be a non-empty object of fields to set.", False
    # Weak models double-wrap: update_spec(patch={"patch": {...real fields...}}),
    # sometimes with sibling fields too. "patch" is the ARGUMENT name, never a
    # spec field — merge the inner dict UP and drop the phantom 'patch' key
    # (rather than erroring on it). Bounded loop handles triple-wraps.
    for _ in range(5):
        if isinstance(patch, dict) and isinstance(patch.get("patch"), dict):
            inner = patch.pop("patch")
            patch = {**inner, **patch}  # any real sibling fields win over inner
        else:
            break
    patch = _coerce_field_shapes(patch)
    shape_error = _validate_field_shapes(patch)
    if shape_error:
        return shape_error, False
    metric_error = _validate_metric_refs(patch)
    if metric_error:
        return metric_error, False
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
    # 'patch' is the update_spec ARGUMENT name, not a spec field — a weak model
    # tries to remove it and gets a misleading "✓ removed". Reject unknown names
    # so the feed is honest and the agent stops chasing a phantom field.
    unknown = [k for k in keys if k not in ALLOWED_TOP_LEVEL]
    if unknown:
        return (
            f"Not spec fields, so nothing to remove: {unknown}. ('patch' is the "
            "update_spec argument name, not a field — the fields go directly "
            f"inside it.) Removable fields: {sorted(ALLOWED_TOP_LEVEL)}.",
            False,
        )
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


def _canonicalize_model(spec: InvestigationSpec) -> InvestigationSpec:
    """Rewrite the spec's model id to its canonical Hugging Face repo.

    A friendly/display name like ``gpt2-small`` is NOT a real Hub repo (the repo
    is ``gpt2``); ``pythia-70m`` is really ``EleutherAI/pythia-70m``. We resolve
    it here, at the single chokepoint before the spec is persisted and executed,
    so the id the agent reads loads directly — even if the agent writes a raw
    ``GPT2LMHeadModel.from_pretrained(<id>)`` instead of going through
    ``load_model()``. Without this, that raw call 404s and a weak driver tends to
    misread the 404 as "no hardware / can't load the model" and asks to revise
    the spec. The original name is kept in ``model.metadata['display_name']``.
    Uses the torch-free alias module so finalize stays cheap.
    """
    from autointerp.tools.model_aliases import resolve_model_id

    canonical = resolve_model_id(spec.model.model_id)
    if canonical == spec.model.model_id:
        return spec
    meta = dict(spec.model.metadata)
    meta.setdefault("display_name", spec.model.model_id)
    return spec.model_copy(
        update={
            "model": spec.model.model_copy(
                update={"model_id": canonical, "metadata": meta}
            )
        }
    )


async def _finalize_spec(args: dict[str, Any]) -> tuple[str, bool]:
    """Lock the plan in. Single call: the agent presents the plan and gets the
    user's approval in conversation FIRST, then calls this once. It re-runs the
    full validation (so it never commits a doomed plan) and writes the approved
    spec; the investigation then launches automatically. There is no second
    "confirm" round — one approval, one finalize."""
    _load_draft_if_present()
    # The approver is the person who said "approve" in chat — i.e. the human
    # user. Default it so the agent NEVER has to ask the user for an "approver"
    # name or "human vs agent" kind (a cryptic internal detail that must not leak
    # into the conversation).
    approver = str(args.get("approver") or "user").strip() or "user"
    kind = str(args.get("approver_kind") or "human").strip()
    if kind not in {"human", "agent"}:
        kind = "human"
    notes = args.get("notes")
    # The user must have approved THIS turn (the REPL gates on it). Drafting and
    # finalizing on the same turn — running a plan the user never saw — is never
    # right. Block it and tell the agent to present and wait. (Draft is kept, so
    # the next call after the user approves succeeds.)
    if _approval_required and not _user_approved:
        return (
            "Not yet — the user has NOT approved this plan. Do not finalize on the "
            "same turn you draft or change it. PRESENT the plan in plain language "
            "(question, model, dataset, stages, metrics, and what would count as "
            "success), then STOP and let them reply. Call finalize_spec only after "
            "they give a go-ahead (\"approve\", \"yes\", \"go\", \"run it\"). Locking "
            "in a plan the user hasn't seen is never correct.",
            False,
        )

    try:
        spec = _partial.try_build(status=SpecStatus.AWAITING_APPROVAL)
    except Exception as exc:
        return f"Cannot finalize — spec has structural errors:\n{exc}", False
    spec = _canonicalize_model(spec)
    # Run the SAME checks validate_spec runs — registered + producible metrics,
    # custom-metric runtime sanity, custom-without-def, and placeholder models —
    # so a plan that validated is a plan that finalizes, and no doomed spec
    # reaches an expensive run. (validate_spec runs these too; keeping them here
    # means a direct finalize that skipped validate is still safe.)
    blockers = _finalize_blockers(spec)
    if blockers:
        return (
            "Cannot finalize — fix each issue (via update_spec) and finalize "
            "again. Call list_metrics to see which metrics are runnable:\n- "
            + "\n- ".join(blockers),
            False,
        )

    SPEC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # If this (spec_id, revision) was already finalized and run once — e.g. the
    # user is re-finalizing after a request_spec_revision — auto-bump to the
    # next free revision. Otherwise we'd silently overwrite the prior spec file:
    # the REPL keys its auto-launch off a NEW spec path, and the run dir for the
    # old revision already exists, so the investigation would never start.
    out_path = SPEC_OUTPUT_DIR / f"{spec.spec_id}_rev{spec.revision}.json"
    if out_path.exists():
        base_rev = spec.revision
        next_rev = base_rev + 1
        while (SPEC_OUTPUT_DIR / f"{spec.spec_id}_rev{next_rev}.json").exists():
            next_rev += 1
        spec = spec.model_copy(update={
            "revision": next_rev,
            "parent_spec_id": spec.parent_spec_id or spec.spec_id,
            "prior_results_ref": (
                spec.prior_results_ref or f"runs/{spec.spec_id}_rev{base_rev}"
            ),
            "revision_reason": spec.revision_reason or "revised after a prior run",
        })
        out_path = SPEC_OUTPUT_DIR / f"{spec.spec_id}_rev{next_rev}.json"
        try:  # the bumped revision must satisfy the revision>1 DAG constraints
            spec = InvestigationSpec.model_validate(spec.model_dump())
        except Exception as exc:  # noqa: BLE001
            return f"Cannot finalize — revised spec is invalid:\n{exc}", False

    approval = make_approval(approver=approver, kind=kind, notes=notes)
    finalized = spec.model_copy(
        update={"status": SpecStatus.APPROVED, "approval": approval}
    )
    out_path.write_text(finalized.model_dump_json(indent=2) + "\n")
    _pending_finalize.clear()
    _clear_draft()
    return (
        f"Plan finalized as revision {finalized.revision}, saved to {out_path}. "
        f"The investigation now runs AUTOMATICALLY and INLINE here (you will see "
        f"its live progress) — it does NOT run in the background and will NOT "
        f"pause for a reply. Your turn is done: write ONE short line confirming "
        f"the plan is locked and the run is starting. Do NOT say it is 'running "
        f"in the background', do NOT ask a question or offer choices, and do NOT "
        f"run experiments or call bash. End your turn now.",
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
            "Set fields of the draft plan: call update_spec(patch={<field>: "
            "<value>, ...}). The fields go DIRECTLY inside `patch` — do NOT nest "
            "another 'patch' key, and 'patch' is not itself a field. Typed fields "
            "are objects: behavior={behavior_id, description}, model={model_id} "
            "(e.g. {'model_id':'gpt2'}), dataset={dataset_id, source, n_samples}, "
            "each stage={stage, pattern, tools, metrics}. A bare string for "
            "behavior/model is auto-wrapped and stray keys are dropped; unknown "
            "field names are rejected. Call describe_spec for the full schema."
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
            "Lock the plan in. Call this ONCE, after you have shown the plan and "
            "the user has approved it — just call finalize_spec() with NO "
            "arguments. The approver is recorded automatically as the human user, "
            "so NEVER ask the user for an 'approver' name or 'human vs agent' "
            "kind; those are internal and must not appear in the conversation. It "
            "re-validates and writes the approved spec; the investigation then "
            "launches automatically. Do not ask for approval a second time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "approver": {
                    "type": "string",
                    "description": "Optional; defaults to the human user. Do not "
                    "ask the user for this.",
                },
                "approver_kind": {"type": "string", "enum": ["human", "agent"]},
                "notes": {"type": "string"},
            },
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
    # Compiling proves it PARSES; dry-run it to prove it RUNS — catches the
    # `sqrt is not defined` class of bug here, not deep inside the investigation.
    from autointerp.pipelines.investigation.metrics import preflight_custom_metric

    runtime_error = preflight_custom_metric(defn)
    if runtime_error:
        return f"Rejected — {runtime_error}", False
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
    global _approval_required, _user_approved
    _partial.reset()
    _pending_finalize.clear()
    _approval_required = False
    _user_approved = False


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
