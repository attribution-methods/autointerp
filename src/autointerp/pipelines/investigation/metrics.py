"""Canonical metric registry — the only legitimate path to a MetricResult.

The investigation pipeline forbids agents from writing ``MetricResult``
artifacts directly. They must call ``compute_metric``, which:

1. dispatches to the canonical implementation for the named ``MetricName``;
2. validates the input shape that implementation declares;
3. computes the value, range-checked against ``METRIC_META`` for that metric;
4. builds a ``MetricResult`` payload (id, value, threshold, pass/fail,
   metadata including unclipped value and inputs_hash);
5. registers a one-time provenance token in ``state.pending_provenance_tokens``
   so ``commit_artifact("MetricResult", ...)`` can verify the agent did not
   tamper with the value before committing.

If an agent wants a metric that isn't here, the right path is a spec revision
(extending ``MetricName`` is a PR-time change, not a runtime one).
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from autointerp.spec import METRIC_META, CustomMetricDef, InvestigationSpec, MetricName

from .run_dir import RunHandle
from .state import ProvenanceToken, now_iso, read_state, write_state


# ---- error types -----------------------------------------------------------


class MetricRegistryError(Exception):
    """Raised when a compute_metric call violates the registry contract."""


# ---- canonical implementations --------------------------------------------


@dataclass(frozen=True)
class MetricImpl:
    name: MetricName
    required_inputs: tuple[str, ...]
    compute: Callable[[dict[str, Any]], float]
    one_line: str


def _require_keys(name: MetricName, inputs: dict[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in inputs]
    if missing:
        raise MetricRegistryError(
            f"{name.value}: missing required input keys: {sorted(missing)}; "
            f"got {sorted(inputs)}"
        )


def _as_float_list(name: MetricName, key: str, raw: Any) -> list[float]:
    if not isinstance(raw, list) or not raw:
        raise MetricRegistryError(f"{name.value}.{key} must be a non-empty list")
    out: list[float] = []
    for i, v in enumerate(raw):
        try:
            out.append(float(v))
        except (TypeError, ValueError) as exc:
            raise MetricRegistryError(
                f"{name.value}.{key}[{i}] is not a number: {v!r}"
            ) from exc
    return out


def _accuracy(inputs: dict[str, Any]) -> float:
    """Mean(predictions[i] == labels[i]).

    Required inputs:
      - predictions: list[Any] of model outputs
      - labels: list[Any] of gold labels (equal length to predictions)
    """
    _require_keys(MetricName.ACCURACY, inputs, ("predictions", "labels"))
    preds = inputs["predictions"]
    labels = inputs["labels"]
    if not isinstance(preds, list) or not isinstance(labels, list):
        raise MetricRegistryError("accuracy: predictions and labels must be lists")
    if len(preds) != len(labels):
        raise MetricRegistryError(
            f"accuracy: predictions ({len(preds)}) and labels ({len(labels)}) "
            "have different lengths"
        )
    if not preds:
        raise MetricRegistryError("accuracy: predictions list is empty")
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(preds)


def _logit_diff(inputs: dict[str, Any]) -> float:
    """Mean(target_logits[i] - foil_logits[i]).

    Required inputs:
      - target_logits: list[float], one per example
      - foil_logits:   list[float], one per example (same length)
    """
    _require_keys(MetricName.LOGIT_DIFF, inputs, ("target_logits", "foil_logits"))
    t = _as_float_list(MetricName.LOGIT_DIFF, "target_logits", inputs["target_logits"])
    f = _as_float_list(MetricName.LOGIT_DIFF, "foil_logits", inputs["foil_logits"])
    if len(t) != len(f):
        raise MetricRegistryError(
            f"logit_diff: target_logits ({len(t)}) and foil_logits ({len(f)}) "
            "have different lengths"
        )
    return sum(a - b for a, b in zip(t, f)) / len(t)


def _faithfulness(inputs: dict[str, Any]) -> float:
    """Wang et al. (2022) IOI faithfulness:

        (circuit_metric - corrupted_metric) / (full_model_metric - corrupted_metric)

    Required inputs:
      - circuit_metric:    float — behavior with rest-of-model ablated
      - full_model_metric: float — clean run
      - corrupted_metric:  float — fully-ablated baseline
    """
    _require_keys(
        MetricName.FAITHFULNESS,
        inputs,
        ("circuit_metric", "full_model_metric", "corrupted_metric"),
    )
    circuit = float(inputs["circuit_metric"])
    full = float(inputs["full_model_metric"])
    corrupted = float(inputs["corrupted_metric"])
    denom = full - corrupted
    if math.isclose(denom, 0.0, abs_tol=1e-9):
        raise MetricRegistryError(
            "faithfulness: full_model_metric and corrupted_metric are equal — "
            "no behavioral gap to recover, faithfulness is undefined"
        )
    return (circuit - corrupted) / denom


def _kl_to_clean(inputs: dict[str, Any]) -> float:
    """Mean per-sample KL(intervened || clean).

    The agent is expected to compute per-sample KL divergences itself
    (vocab-sized distributions don't belong in the registry's input dict).
    The canonical aggregation here is: validate non-negativity, take the
    mean.

    Required inputs:
      - kl_per_sample: list[float], one non-negative KL value per example
    """
    _require_keys(MetricName.KL_TO_CLEAN, inputs, ("kl_per_sample",))
    vals = _as_float_list(MetricName.KL_TO_CLEAN, "kl_per_sample", inputs["kl_per_sample"])
    for i, v in enumerate(vals):
        if v < 0 or math.isnan(v):
            raise MetricRegistryError(
                f"kl_to_clean.kl_per_sample[{i}] = {v!r} is invalid; "
                "KL divergence must be a finite, non-negative number"
            )
    return sum(vals) / len(vals)


def _patch_effect_recovery(inputs: dict[str, Any]) -> float:
    """Fraction of the clean-vs-corrupted gap recovered by patching:

        recovery = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)

    Required inputs:
      - clean_metric:   float — behavior on the clean run
      - corrupt_metric: float — behavior on the fully-corrupted run
      - patched_metric: float — behavior with the candidate patch applied
    """
    _require_keys(
        MetricName.PATCH_EFFECT_RECOVERY,
        inputs,
        ("clean_metric", "corrupt_metric", "patched_metric"),
    )
    clean = float(inputs["clean_metric"])
    corrupt = float(inputs["corrupt_metric"])
    patched = float(inputs["patched_metric"])
    denom = clean - corrupt
    if math.isclose(denom, 0.0, abs_tol=1e-9):
        raise MetricRegistryError(
            "patch_effect_recovery: clean_metric and corrupt_metric are equal — "
            "no behavioral gap to recover; recovery is undefined"
        )
    return (patched - corrupt) / denom


def _ablation_drop(inputs: dict[str, Any]) -> float:
    """Drop in a behavioral metric when a candidate component is ablated:

        drop = baseline_metric - ablated_metric

    Sign convention: positive drop = ablating the component hurt the
    behavior (component was necessary). Range is unbounded (negative
    values are allowed and meaningful — they say the ablation *helped*).

    Required inputs:
      - baseline_metric: float — metric value on the unablated run
      - ablated_metric:  float — metric value with the component ablated
    """
    _require_keys(
        MetricName.ABLATION_DROP,
        inputs,
        ("baseline_metric", "ablated_metric"),
    )
    baseline = float(inputs["baseline_metric"])
    ablated = float(inputs["ablated_metric"])
    return baseline - ablated


def _effect_size(inputs: dict[str, Any]) -> float:
    """Cohen's d between two distributions.

    Unpaired form: d = (mean(group_a) - mean(group_b)) / pooled_sd, where
    pooled_sd uses the sample SD (ddof=1) and equal-weight pooling. Sign
    follows group_a - group_b: positive d means group_a is larger on average.

    Paired form (when ``paired=True`` is set in inputs): d_z = mean(diff) /
    sd(diff), where diff = group_a[i] - group_b[i].

    Required inputs:
      - group_a: list[float] — observations from condition A
      - group_b: list[float] — observations from condition B
    Optional inputs:
      - paired: bool — if True, compute Cohen's d_z; group_a and group_b
        must then have equal length
    """
    _require_keys(MetricName.EFFECT_SIZE, inputs, ("group_a", "group_b"))
    a = _as_float_list(MetricName.EFFECT_SIZE, "group_a", inputs["group_a"])
    b = _as_float_list(MetricName.EFFECT_SIZE, "group_b", inputs["group_b"])
    paired = bool(inputs.get("paired", False))

    if paired:
        if len(a) != len(b):
            raise MetricRegistryError(
                f"effect_size: paired=True requires equal lengths "
                f"(got {len(a)} vs {len(b)})"
            )
        if len(a) < 2:
            raise MetricRegistryError(
                "effect_size: paired form needs at least 2 observations"
            )
        diffs = [ai - bi for ai, bi in zip(a, b)]
        m = sum(diffs) / len(diffs)
        var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
        sd = math.sqrt(var)
        if sd == 0.0:
            raise MetricRegistryError(
                "effect_size: paired diffs have zero SD; effect size undefined"
            )
        return m / sd

    if len(a) < 2 or len(b) < 2:
        raise MetricRegistryError(
            "effect_size: each group needs at least 2 observations"
        )
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    var_a = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    sd = math.sqrt(pooled)
    if sd == 0.0:
        raise MetricRegistryError(
            "effect_size: pooled SD is zero; effect size undefined"
        )
    return (ma - mb) / sd


def _auroc(inputs: dict[str, Any]) -> float:
    """Binary ROC-AUC for ``scores`` against 0/1 ``labels``.

    Computed by the rank-sum identity (Mann-Whitney U): with no ties,
    AUC = (sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg).
    Tied scores are handled by midranks. Range [0, 1]; 0.5 = chance.

    Required inputs:
      - scores: list[float] — model scores (higher = positive)
      - labels: list[int]   — 0/1 labels parallel to scores
    """
    _require_keys(MetricName.AUROC, inputs, ("scores", "labels"))
    scores = _as_float_list(MetricName.AUROC, "scores", inputs["scores"])
    raw_labels = inputs["labels"]
    if not isinstance(raw_labels, list) or len(raw_labels) != len(scores):
        raise MetricRegistryError(
            f"auroc: labels must be a list parallel to scores "
            f"({len(scores)} scores)"
        )
    labels: list[int] = []
    for i, l in enumerate(raw_labels):
        if l in (0, 1, True, False):
            labels.append(int(bool(l)))
        else:
            raise MetricRegistryError(
                f"auroc.labels[{i}] = {l!r} is not 0/1 (or boolean)"
            )
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise MetricRegistryError(
            "auroc: need at least one positive and one negative label"
        )

    # Midrank assignment (for ties): sort by score, assign average of the
    # tied positions as the rank.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed midrank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(r for r, l in zip(ranks, labels) if l == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _hit_rate(inputs: dict[str, Any]) -> float:
    """Fraction of trials satisfying a behavioral predicate.

    The agent evaluates the predicate in its own script (predicates can't
    be JSON-serialized) and passes the boolean result vector.

    Required inputs:
      - hits: list[bool|int] — 1 if the predicate fired, 0 otherwise
    """
    _require_keys(MetricName.HIT_RATE, inputs, ("hits",))
    raw = inputs["hits"]
    if not isinstance(raw, list) or not raw:
        raise MetricRegistryError("hit_rate.hits must be a non-empty list")
    hits = 0
    for i, v in enumerate(raw):
        if v in (0, 1, True, False):
            hits += int(bool(v))
        else:
            raise MetricRegistryError(
                f"hit_rate.hits[{i}] = {v!r} is not 0/1 (or boolean)"
            )
    return hits / len(raw)


def _necessity_drop(inputs: dict[str, Any]) -> float:
    """Drop in a behavioral metric when a component is removed in vivo:

        necessity_drop = clean_metric - with_component_removed

    Sign convention: positive drop = component is necessary (removal hurts).

    Required inputs:
      - clean_metric: float — full-model behavior
      - with_component_removed: float — full model with the component ablated
    """
    _require_keys(
        MetricName.NECESSITY_DROP,
        inputs,
        ("clean_metric", "with_component_removed"),
    )
    clean = float(inputs["clean_metric"])
    removed = float(inputs["with_component_removed"])
    return clean - removed


def _completeness(inputs: dict[str, Any]) -> float:
    """Behavior reproduced when only the circuit is intact:

        completeness = circuit_only_metric / full_model_metric

    Range [0, 1] (range-clipping in the registry preserves the unclipped
    value in metadata if the agent passes a circuit_only > full).

    Required inputs:
      - full_model_metric:    float — clean full-model behavior
      - circuit_only_metric:  float — behavior with everything-but-circuit ablated
    """
    _require_keys(
        MetricName.COMPLETENESS,
        inputs,
        ("full_model_metric", "circuit_only_metric"),
    )
    full = float(inputs["full_model_metric"])
    circuit_only = float(inputs["circuit_only_metric"])
    if math.isclose(full, 0.0, abs_tol=1e-9):
        raise MetricRegistryError(
            "completeness: full_model_metric is 0; ratio is undefined"
        )
    return circuit_only / full


def _sufficiency(inputs: dict[str, Any]) -> float:
    """Behavior reproduced when only the candidate component is active:

        sufficiency = only_component_metric / full_model_metric

    Range [0, 1]. Same shape as completeness but for a single component.

    Required inputs:
      - full_model_metric:      float — clean full-model behavior
      - only_component_metric:  float — behavior with everything else ablated
    """
    _require_keys(
        MetricName.SUFFICIENCY,
        inputs,
        ("full_model_metric", "only_component_metric"),
    )
    full = float(inputs["full_model_metric"])
    only = float(inputs["only_component_metric"])
    if math.isclose(full, 0.0, abs_tol=1e-9):
        raise MetricRegistryError(
            "sufficiency: full_model_metric is 0; ratio is undefined"
        )
    return only / full


def _minimality(inputs: dict[str, Any]) -> float:
    """Smallest single-component faithfulness drop in the circuit.

    For each component in the final circuit, the agent measures
    faithfulness with that component removed. Minimality = the *minimum*
    drop across the sweep (= worst-case "did this component pull
    weight?"). A circuit passes a `minimality >= τ` criterion iff every
    component contributes at least a τ-sized drop. The full distribution
    is preserved in metadata for inspection.

    Required inputs:
      - full_circuit_faithfulness: float — faithfulness of the full
        candidate circuit
      - removed_faithfulness: list[float] — faithfulness with each
        single component removed (one entry per component)
    """
    _require_keys(
        MetricName.MINIMALITY,
        inputs,
        ("full_circuit_faithfulness", "removed_faithfulness"),
    )
    full = float(inputs["full_circuit_faithfulness"])
    removed = _as_float_list(
        MetricName.MINIMALITY, "removed_faithfulness", inputs["removed_faithfulness"]
    )
    drops = [full - r for r in removed]
    # Negative drops mean "removing the component *helped* faithfulness" —
    # the component is unnecessary. Surface the worst case directly so the
    # threshold check has the semantics "every component drops by at least τ".
    return min(drops)


REGISTRY: dict[MetricName, MetricImpl] = {
    MetricName.ACCURACY: MetricImpl(
        name=MetricName.ACCURACY,
        required_inputs=("predictions", "labels"),
        compute=_accuracy,
        one_line="Mean(predictions == labels).",
    ),
    MetricName.LOGIT_DIFF: MetricImpl(
        name=MetricName.LOGIT_DIFF,
        required_inputs=("target_logits", "foil_logits"),
        compute=_logit_diff,
        one_line="Mean(target_logits - foil_logits).",
    ),
    MetricName.FAITHFULNESS: MetricImpl(
        name=MetricName.FAITHFULNESS,
        required_inputs=("circuit_metric", "full_model_metric", "corrupted_metric"),
        compute=_faithfulness,
        one_line="(circuit - corrupted) / (full - corrupted).",
    ),
    MetricName.KL_TO_CLEAN: MetricImpl(
        name=MetricName.KL_TO_CLEAN,
        required_inputs=("kl_per_sample",),
        compute=_kl_to_clean,
        one_line="Mean(kl_per_sample); per-sample KL(intervened || clean).",
    ),
    MetricName.PATCH_EFFECT_RECOVERY: MetricImpl(
        name=MetricName.PATCH_EFFECT_RECOVERY,
        required_inputs=("clean_metric", "corrupt_metric", "patched_metric"),
        compute=_patch_effect_recovery,
        one_line="(patched - corrupt) / (clean - corrupt).",
    ),
    MetricName.ABLATION_DROP: MetricImpl(
        name=MetricName.ABLATION_DROP,
        required_inputs=("baseline_metric", "ablated_metric"),
        compute=_ablation_drop,
        one_line="baseline_metric - ablated_metric.",
    ),
    MetricName.MINIMALITY: MetricImpl(
        name=MetricName.MINIMALITY,
        required_inputs=("full_circuit_faithfulness", "removed_faithfulness"),
        compute=_minimality,
        one_line="min(full_circuit_faithfulness - removed_faithfulness[i]).",
    ),
    MetricName.EFFECT_SIZE: MetricImpl(
        name=MetricName.EFFECT_SIZE,
        required_inputs=("group_a", "group_b"),
        compute=_effect_size,
        one_line="Cohen's d (or paired d_z) between group_a and group_b.",
    ),
    MetricName.AUROC: MetricImpl(
        name=MetricName.AUROC,
        required_inputs=("scores", "labels"),
        compute=_auroc,
        one_line="Binary ROC-AUC via Mann-Whitney U with midrank tie handling.",
    ),
    MetricName.HIT_RATE: MetricImpl(
        name=MetricName.HIT_RATE,
        required_inputs=("hits",),
        compute=_hit_rate,
        one_line="Mean(hits); fraction of trials satisfying the predicate.",
    ),
    MetricName.NECESSITY_DROP: MetricImpl(
        name=MetricName.NECESSITY_DROP,
        required_inputs=("clean_metric", "with_component_removed"),
        compute=_necessity_drop,
        one_line="clean_metric - with_component_removed.",
    ),
    MetricName.COMPLETENESS: MetricImpl(
        name=MetricName.COMPLETENESS,
        required_inputs=("full_model_metric", "circuit_only_metric"),
        compute=_completeness,
        one_line="circuit_only_metric / full_model_metric, clipped to [0, 1].",
    ),
    MetricName.SUFFICIENCY: MetricImpl(
        name=MetricName.SUFFICIENCY,
        required_inputs=("full_model_metric", "only_component_metric"),
        compute=_sufficiency,
        one_line="only_component_metric / full_model_metric, clipped to [0, 1].",
    ),
}


# ---- range handling -------------------------------------------------------


def _clip_to_range(
    metric: MetricName,
    value: float,
    *,
    custom_def: CustomMetricDef | None = None,
) -> tuple[float, bool]:
    """Clip to METRIC_META range (or custom def range); return (value, clipped)."""
    if custom_def is not None:
        lo, hi = custom_def.value_range
    else:
        meta = METRIC_META.get(metric)
        if meta is None:
            return value, False
        lo, hi = meta.value_range
    clipped = value
    if lo is not None and clipped < lo:
        clipped = lo
    if hi is not None and clipped > hi:
        clipped = hi
    return clipped, clipped != value


# ---- custom-metric dispatch -----------------------------------------------


_SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
    "range": range, "sorted": sorted, "enumerate": enumerate, "zip": zip,
    "any": any, "all": all, "round": round, "float": float, "int": int,
    "bool": bool, "str": str, "list": list, "tuple": tuple, "dict": dict,
    "isinstance": isinstance, "map": map, "filter": filter,
}


def _load_run_spec(handle: RunHandle) -> InvestigationSpec:
    return InvestigationSpec.model_validate_json(handle.spec_path.read_text())


def _resolve_custom_def(handle: RunHandle, inputs: dict[str, Any]) -> CustomMetricDef:
    name = inputs.get("__custom_name__")
    if not isinstance(name, str) or not name:
        raise MetricRegistryError(
            "metric=custom requires `inputs['__custom_name__']` (the name of "
            "the custom metric defined in the spec)"
        )
    spec = _load_run_spec(handle)
    for crit in spec.success_criteria:
        if (
            crit.metric == MetricName.CUSTOM
            and crit.custom_metric_def is not None
            and crit.custom_metric_def.name == name
        ):
            return crit.custom_metric_def
    available = [
        c.custom_metric_def.name
        for c in spec.success_criteria
        if c.custom_metric_def is not None
    ]
    raise MetricRegistryError(
        f"unknown custom metric {name!r}; spec defines: {available}"
    )


def _custom_impl(defn: CustomMetricDef) -> "MetricImpl":
    """Compile the custom metric source and wrap as a MetricImpl."""
    namespace: dict[str, Any] = {}
    safe_globals = {"__builtins__": _SAFE_BUILTINS, "math": math}
    try:
        exec(compile(defn.source_code, f"<custom:{defn.name}>", "exec"),
             safe_globals, namespace)
    except Exception as exc:
        raise MetricRegistryError(
            f"custom metric {defn.name!r} failed to compile: {exc}"
        ) from exc
    fn = namespace.get(defn.function_name)
    if fn is None or not callable(fn):
        raise MetricRegistryError(
            f"custom metric {defn.name!r}: source did not define callable "
            f"{defn.function_name!r}"
        )

    def _wrapped(inputs: dict[str, Any]) -> float:
        missing = [k for k in defn.requires_inputs if k not in inputs]
        if missing:
            raise MetricRegistryError(
                f"custom metric {defn.name!r}: missing required inputs {missing}"
            )
        try:
            return float(fn(inputs))
        except MetricRegistryError:
            raise
        except Exception as exc:
            raise MetricRegistryError(
                f"custom metric {defn.name!r} raised at runtime: {exc}"
            ) from exc

    return MetricImpl(
        name=MetricName.CUSTOM,
        required_inputs=tuple(defn.requires_inputs),
        compute=_wrapped,
        one_line=defn.description,
    )


# ---- hashing & token issuance ---------------------------------------------


def _hash_inputs(inputs: dict[str, Any]) -> str:
    """Stable sha256 over a normalized JSON dump of the inputs."""
    blob = json.dumps(inputs, sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _new_token() -> str:
    return "tok_" + secrets.token_hex(16)


# ---- evaluation against threshold ------------------------------------------


def _evaluate_threshold(
    value: float, threshold: float | None, comparator: str | None
) -> bool | None:
    """Match the comparator semantics used by ``Criterion``.

    The agent may pass an optional ``threshold`` and ``comparator`` so the
    resulting MetricResult records pass/fail for downstream display. Real
    success-criterion evaluation is the job of ``criteria.evaluate_criterion``,
    not this function.
    """
    if threshold is None or comparator is None:
        return None
    if comparator == ">=":
        return value >= threshold
    if comparator == ">":
        return value > threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "==":
        return value == threshold
    raise MetricRegistryError(f"unknown comparator {comparator!r}")


# ---- public entry point ----------------------------------------------------


def compute_metric(
    handle: RunHandle,
    *,
    metric: MetricName | str,
    metric_id: str,
    inputs: dict[str, Any],
    threshold: float | None = None,
    comparator: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Run the canonical implementation for ``metric`` and issue a one-time token.

    Returns a tuple ``(metric_result_payload, provenance_token)``. The agent
    is expected to immediately ``commit_artifact("MetricResult", payload,
    split=..., provenance_token=token)`` — the token is single-use and tied
    to the current stage.
    """
    # Coerce string -> enum so the agent can pass either.
    if isinstance(metric, str):
        try:
            metric = MetricName(metric)
        except ValueError as exc:
            raise MetricRegistryError(
                f"unknown metric {metric!r}; valid: {[m.value for m in MetricName]}"
            ) from exc

    if not isinstance(inputs, dict):
        raise MetricRegistryError("inputs must be a dict")
    if not metric_id or not isinstance(metric_id, str):
        raise MetricRegistryError("metric_id must be a non-empty string")

    custom_def = None
    if metric == MetricName.CUSTOM:
        custom_def = _resolve_custom_def(handle, inputs)
        impl = _custom_impl(custom_def)
        # Drop the lookup-only key so it isn't part of the inputs hash.
        inputs = {k: v for k, v in inputs.items() if k != "__custom_name__"}
    else:
        impl = REGISTRY.get(metric)
        if impl is None:
            raise MetricRegistryError(
                f"no canonical implementation for {metric.value!r}; "
                f"registered: {[m.value for m in REGISTRY]}"
            )

    from .guards import GuardError, enforce_budget

    try:
        enforce_budget(handle)
    except GuardError as exc:
        raise MetricRegistryError(str(exc)) from exc

    state = read_state(handle.state_path)
    if state.terminal_state is not None:
        raise MetricRegistryError(
            f"run is in terminal state {state.terminal_state.value!r}; "
            "no further metric computations"
        )
    stage_idx = state.current_stage_idx

    # Validate + compute.
    raw_value = float(impl.compute(inputs))
    clipped_value, was_clipped = _clip_to_range(metric, raw_value, custom_def=custom_def)
    inputs_hash = _hash_inputs(inputs)

    md: dict[str, Any] = dict(metadata or {})
    if custom_def is not None:
        md.update(
            {
                "metric_name": custom_def.name,
                "metric_family": custom_def.family.value,
                "inputs_hash": inputs_hash,
                "registry_version": 1,
                "custom": True,
                "source_hash": custom_def.source_hash,
            }
        )
    else:
        md.update(
            {
                "metric_name": metric.value,
                "metric_family": METRIC_META[metric].family.value,
                "inputs_hash": inputs_hash,
                "registry_version": 1,
            }
        )
    if was_clipped:
        md["unclipped_value"] = raw_value
        md["clipped_to_range"] = list(METRIC_META[metric].value_range) if custom_def is None else list(custom_def.value_range)

    passed = _evaluate_threshold(clipped_value, threshold, comparator)

    payload: dict[str, Any] = {
        "metric_id": metric_id,
        "value": clipped_value,
        "threshold": threshold,
        "passed": passed,
        "metadata": md,
    }

    # Issue the one-time token and persist.
    token = _new_token()
    issued_at = now_iso()
    state.pending_provenance_tokens[token] = ProvenanceToken(
        metric=metric.value,
        metric_id=metric_id,
        value=clipped_value,
        inputs_hash=inputs_hash,
        issued_at=issued_at,
        stage_idx=stage_idx,
    )
    state.budget_consumed.tool_calls += 1
    write_state(handle.state_path, state)

    log_entry = {
        "ts": issued_at,
        "stage_idx": stage_idx,
        "tool": "compute_metric",
        "args": {
            "metric": metric.value,
            "metric_id": metric_id,
            "inputs_hash": inputs_hash,
        },
        "ok": True,
        "result_summary": f"value={clipped_value!r}; token={token}",
        "budget_after": state.budget_consumed.model_dump(),
    }
    line = json.dumps(log_entry, separators=(",", ":")) + "\n"
    with handle.log_path.open("a") as fh:
        fh.write(line)

    return payload, token


def compute_and_commit_metric(
    handle: RunHandle,
    *,
    metric: MetricName | str,
    metric_id: str,
    inputs: dict[str, Any],
    split: str,
    threshold: float | None = None,
    comparator: str | None = None,
    metadata: dict[str, Any] | None = None,
    criterion_id: str | None = None,
    inconclusive_reason: str | None = None,
) -> dict[str, Any]:
    """One-shot metric flow: compute, commit, optionally evaluate a criterion.

    Replaces the recipe that was being reinvented per run as ``stage0_*.py``
    scripts: ``compute_metric`` → ``commit_artifact("MetricResult", ...,
    provenance_token=...)`` → ``evaluate_criterion`` (when ``criterion_id``
    is given). All three steps share a single budget tick worth's of UX
    weight; rolling them into one tool removes ~3 agent iterations per
    metric, and removes a class of "agent forgot to commit" bugs.

    Returns a dict with keys ``metric_result`` (the validated payload),
    ``artifact_ref`` (kind/relpath/stage_idx/split), and ``criterion_record``
    (the CriterionRecord if ``criterion_id`` was given, else None).

    ``inconclusive_reason`` is forwarded to ``evaluate_criterion``: when set
    (and ``criterion_id`` given) the criterion is recorded INCONCLUSIVE
    instead of PASS/FAIL and the run is not terminated.
    """
    from .artifacts import commit_artifact  # local import to avoid cycles
    from .criteria import evaluate_criterion

    payload, token = compute_metric(
        handle,
        metric=metric,
        metric_id=metric_id,
        inputs=inputs,
        threshold=threshold,
        comparator=comparator,
        metadata=metadata,
    )
    ref = commit_artifact(
        handle, "MetricResult", payload, split=split, provenance_token=token
    )
    out: dict[str, Any] = {
        "metric_result": payload,
        "artifact_ref": {
            "kind": ref.kind,
            "artifact_id": ref.artifact_id,
            "relpath": ref.relpath,
            "stage_idx": ref.stage_idx,
            "split": ref.split,
        },
        "criterion_record": None,
    }
    if criterion_id is not None:
        rec = evaluate_criterion(
            handle,
            criterion_id,
            metric_result_ref=ref.relpath,
            inconclusive_reason=inconclusive_reason,
        )
        out["criterion_record"] = {"criterion_id": criterion_id, **rec.model_dump()}
    return out


__all__ = [
    "MetricImpl",
    "MetricRegistryError",
    "REGISTRY",
    "compute_metric",
    "compute_and_commit_metric",
]
