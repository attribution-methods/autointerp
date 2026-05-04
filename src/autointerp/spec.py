"""Stage 0 investigation specification.

A spec is the pre-registered contract for an investigation. The agent drafts it,
a critic challenges it, a human (later: agent) approves it, and downstream
stages refuse to run without one. Specs form a DAG: a revision points back to
its parent and the prior results that motivated the change.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from autointerp.schemas import (
    BehaviorSpec,
    InvestigationStage,
    ModelRef,
    StrictBaseModel,
)


class SpecStatus(str, Enum):
    DRAFT = "draft"
    UNDER_CRITIQUE = "under_critique"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class InvestigationOutcome(str, Enum):
    """Why the investigation ended. Distinct from per-spec pass/fail."""

    HYPOTHESIS_CONFIRMED = "hypothesis_confirmed"
    HYPOTHESIS_FALSIFIED = "hypothesis_falsified"
    INCONCLUSIVE = "inconclusive"
    ABORTED = "aborted"
    PENDING = "pending"


class MetricFamily(str, Enum):
    """What kind of evidence the metric provides."""

    BEHAVIORAL = "behavioral"
    LOCALIZATION = "localization"
    CAUSAL = "causal"
    GENERALIZATION = "generalization"
    FEATURE = "feature"
    INFORMATION = "information"


class MetricName(str, Enum):
    """Closed vocabulary for stage metrics. Extend by PR, not at runtime."""

    AUROC = "auroc"
    ACCURACY = "accuracy"
    LOGIT_DIFF = "logit_diff"
    KL_TO_CLEAN = "kl_to_clean"
    FAITHFULNESS = "faithfulness"
    COMPLETENESS = "completeness"
    MINIMALITY = "minimality"
    NECESSITY_DROP = "necessity_drop"
    SUFFICIENCY = "sufficiency"
    PATCH_EFFECT_RECOVERY = "patch_effect_recovery"
    HIT_RATE = "hit_rate"
    EFFECT_SIZE = "effect_size"
    DIRECT_CONTRIBUTION = "direct_contribution"
    ABLATION_DROP = "ablation_drop"
    PARAPHRASE_INVARIANCE = "paraphrase_invariance"
    FEATURE_ACTIVATION_DENSITY = "feature_activation_density"
    FEATURE_LABEL_QUALITY = "feature_label_quality"
    STEERING_EFFECT_SIZE = "steering_effect_size"
    MUTUAL_INFORMATION = "mutual_information"
    CAUSAL_INDIRECT_EFFECT = "causal_indirect_effect"
    CUSTOM = "custom"


class ToolName(str, Enum):
    """Closed vocabulary for tools the agent may invoke at each stage."""

    BLACKBOX_PROBE = "blackbox_probe"
    LINEAR_PROBE = "linear_probe"
    LOGIT_LENS = "logit_lens"
    DLA = "direct_logit_attribution"
    ACTIVATION_PATCHING = "activation_patching"
    PATH_PATCHING = "path_patching"
    ATTRIBUTION_PATCHING = "attribution_patching"
    SAE_INSPECT = "sae_inspect"
    STEERING = "steering"
    CONTRASTIVE_DIRECTIONS = "contrastive_directions"


class MetricMeta(StrictBaseModel):
    """Typed contract for a metric. Used by the validator and the agent picker."""

    name: MetricName
    family: MetricFamily
    value_range: tuple[float | None, float | None]
    direction: Literal["higher", "lower", "either"]
    requires_inputs: list[str] = Field(default_factory=list)
    allowed_params: list[str] = Field(default_factory=list)
    one_line: str = ""


class ToolMeta(StrictBaseModel):
    """Typed contract for a tool. Declares what spec fields it depends on."""

    name: ToolName
    requires_fields: list[str] = Field(default_factory=list)
    families_emitted: list[MetricFamily] = Field(default_factory=list)
    one_line: str = ""


METRIC_META: dict[MetricName, MetricMeta] = {
    MetricName.AUROC: MetricMeta(
        name=MetricName.AUROC, family=MetricFamily.BEHAVIORAL,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["scores", "labels"],
        one_line="ROC AUC for a binary readout (e.g. probe).",
    ),
    MetricName.ACCURACY: MetricMeta(
        name=MetricName.ACCURACY, family=MetricFamily.BEHAVIORAL,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["predictions", "labels"],
        one_line="Top-1 correctness rate.",
    ),
    MetricName.LOGIT_DIFF: MetricMeta(
        name=MetricName.LOGIT_DIFF, family=MetricFamily.BEHAVIORAL,
        value_range=(None, None), direction="higher",
        requires_inputs=["logits", "target_token", "foil_token"],
        one_line="logit(target) - logit(foil) at the prediction position.",
    ),
    MetricName.HIT_RATE: MetricMeta(
        name=MetricName.HIT_RATE, family=MetricFamily.BEHAVIORAL,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["outputs", "predicate"],
        one_line="Fraction of outputs satisfying a behavioral predicate.",
    ),
    MetricName.EFFECT_SIZE: MetricMeta(
        name=MetricName.EFFECT_SIZE, family=MetricFamily.BEHAVIORAL,
        value_range=(None, None), direction="higher",
        requires_inputs=["group_a", "group_b"],
        one_line="Cohen's d (or comparable) between two distributions.",
    ),
    MetricName.KL_TO_CLEAN: MetricMeta(
        name=MetricName.KL_TO_CLEAN, family=MetricFamily.LOCALIZATION,
        value_range=(0.0, None), direction="lower",
        requires_inputs=["p_clean", "p_intervened"],
        one_line="KL divergence between intervened and clean output distributions.",
    ),
    MetricName.DIRECT_CONTRIBUTION: MetricMeta(
        name=MetricName.DIRECT_CONTRIBUTION, family=MetricFamily.LOCALIZATION,
        value_range=(None, None), direction="higher",
        requires_inputs=["component_output", "unembed_direction"],
        one_line="Direct logit-attribution score for a single component.",
    ),
    MetricName.ABLATION_DROP: MetricMeta(
        name=MetricName.ABLATION_DROP, family=MetricFamily.LOCALIZATION,
        value_range=(None, None), direction="higher",
        requires_inputs=["clean_metric", "ablated_metric"],
        one_line="Drop in a behavioral metric when a component is ablated.",
    ),
    MetricName.PATCH_EFFECT_RECOVERY: MetricMeta(
        name=MetricName.PATCH_EFFECT_RECOVERY, family=MetricFamily.CAUSAL,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["clean_metric", "corrupt_metric", "patched_metric"],
        one_line="Fraction of clean-vs-corrupt gap recovered by patching k components.",
    ),
    MetricName.FAITHFULNESS: MetricMeta(
        name=MetricName.FAITHFULNESS, family=MetricFamily.CAUSAL,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["circuit_metric", "full_model_metric"],
        one_line="Behavior reproduced by the candidate circuit alone (Wang et al.).",
    ),
    MetricName.COMPLETENESS: MetricMeta(
        name=MetricName.COMPLETENESS, family=MetricFamily.CAUSAL,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["full_model_metric", "circuit_only_metric"],
        one_line="Behavior reproduced when only the circuit is intact (rest ablated).",
    ),
    MetricName.MINIMALITY: MetricMeta(
        name=MetricName.MINIMALITY, family=MetricFamily.CAUSAL,
        value_range=(0.0, None), direction="higher",
        requires_inputs=["faithfulness_full", "faithfulness_minus_one"],
        one_line="Faithfulness drop when removing any single circuit component.",
    ),
    MetricName.NECESSITY_DROP: MetricMeta(
        name=MetricName.NECESSITY_DROP, family=MetricFamily.CAUSAL,
        value_range=(None, None), direction="higher",
        requires_inputs=["clean_metric", "with_component_removed"],
        one_line="Drop in behavior when a component is removed from the full model.",
    ),
    MetricName.SUFFICIENCY: MetricMeta(
        name=MetricName.SUFFICIENCY, family=MetricFamily.CAUSAL,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["only_component_metric", "full_model_metric"],
        one_line="Behavior reproduced when only the candidate component is active.",
    ),
    MetricName.PARAPHRASE_INVARIANCE: MetricMeta(
        name=MetricName.PARAPHRASE_INVARIANCE, family=MetricFamily.GENERALIZATION,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["original_metric", "paraphrase_metric"],
        one_line="Behavior preserved under prompt paraphrase.",
    ),
    MetricName.FEATURE_ACTIVATION_DENSITY: MetricMeta(
        name=MetricName.FEATURE_ACTIVATION_DENSITY, family=MetricFamily.FEATURE,
        value_range=(0.0, 1.0), direction="either",
        requires_inputs=["feature_activations"],
        one_line="Fraction of tokens at which the SAE feature is active.",
    ),
    MetricName.FEATURE_LABEL_QUALITY: MetricMeta(
        name=MetricName.FEATURE_LABEL_QUALITY, family=MetricFamily.FEATURE,
        value_range=(0.0, 1.0), direction="higher",
        requires_inputs=["label", "judge_score"],
        one_line="Auto-interp judge agreement that the label predicts feature firing.",
    ),
    MetricName.STEERING_EFFECT_SIZE: MetricMeta(
        name=MetricName.STEERING_EFFECT_SIZE, family=MetricFamily.FEATURE,
        value_range=(None, None), direction="higher",
        requires_inputs=["baseline_metric", "steered_metric"],
        one_line="Behavior change when steering with a feature/direction.",
    ),
    MetricName.MUTUAL_INFORMATION: MetricMeta(
        name=MetricName.MUTUAL_INFORMATION, family=MetricFamily.INFORMATION,
        value_range=(0.0, None), direction="higher",
        requires_inputs=["x", "y"],
        one_line="Estimated mutual information between activations and a label.",
    ),
    MetricName.CAUSAL_INDIRECT_EFFECT: MetricMeta(
        name=MetricName.CAUSAL_INDIRECT_EFFECT, family=MetricFamily.CAUSAL,
        value_range=(None, None), direction="higher",
        requires_inputs=["mediated_effect", "total_effect"],
        one_line="Pearl-style indirect effect mediated through a component.",
    ),
    MetricName.CUSTOM: MetricMeta(
        name=MetricName.CUSTOM, family=MetricFamily.BEHAVIORAL,
        value_range=(None, None), direction="either",
        one_line="Custom user-defined metric. Requires `custom_metric_def`. Triggers human review.",
    ),
}


TOOL_META: dict[ToolName, ToolMeta] = {
    ToolName.BLACKBOX_PROBE: ToolMeta(
        name=ToolName.BLACKBOX_PROBE, requires_fields=[],
        families_emitted=[MetricFamily.BEHAVIORAL],
        one_line="Run the model on a prompt batch and score outputs.",
    ),
    ToolName.LINEAR_PROBE: ToolMeta(
        name=ToolName.LINEAR_PROBE, requires_fields=[],
        families_emitted=[MetricFamily.BEHAVIORAL, MetricFamily.LOCALIZATION],
        one_line="Train a linear classifier on cached activations.",
    ),
    ToolName.LOGIT_LENS: ToolMeta(
        name=ToolName.LOGIT_LENS, requires_fields=[],
        families_emitted=[MetricFamily.LOCALIZATION],
        one_line="Project intermediate residual streams through the unembed.",
    ),
    ToolName.DLA: ToolMeta(
        name=ToolName.DLA, requires_fields=[],
        families_emitted=[MetricFamily.LOCALIZATION],
        one_line="Direct logit attribution per component.",
    ),
    ToolName.ACTIVATION_PATCHING: ToolMeta(
        name=ToolName.ACTIVATION_PATCHING, requires_fields=["contrast"],
        families_emitted=[MetricFamily.CAUSAL],
        one_line="Patch a component's activation from clean run into corrupted run.",
    ),
    ToolName.PATH_PATCHING: ToolMeta(
        name=ToolName.PATH_PATCHING, requires_fields=["contrast"],
        families_emitted=[MetricFamily.CAUSAL],
        one_line="Patch a specific source->target path; isolates direct vs indirect effects.",
    ),
    ToolName.ATTRIBUTION_PATCHING: ToolMeta(
        name=ToolName.ATTRIBUTION_PATCHING, requires_fields=["contrast"],
        families_emitted=[MetricFamily.CAUSAL],
        one_line="Gradient-based approximation of activation patching.",
    ),
    ToolName.SAE_INSPECT: ToolMeta(
        name=ToolName.SAE_INSPECT, requires_fields=[],
        families_emitted=[MetricFamily.FEATURE],
        one_line="Decode activations through a sparse autoencoder and label features.",
    ),
    ToolName.STEERING: ToolMeta(
        name=ToolName.STEERING, requires_fields=[],
        families_emitted=[MetricFamily.FEATURE, MetricFamily.CAUSAL],
        one_line="Add a direction to the residual stream and measure behavior change.",
    ),
    ToolName.CONTRASTIVE_DIRECTIONS: ToolMeta(
        name=ToolName.CONTRASTIVE_DIRECTIONS, requires_fields=["contrast"],
        families_emitted=[MetricFamily.LOCALIZATION, MetricFamily.FEATURE],
        one_line="Find directions distinguishing positive vs negative populations.",
    ),
}


class PatternId(str, Enum):
    """Named investigation patterns. `custom` triggers stronger review."""

    BLACKBOX_THEN_PATCHING = "blackbox_then_patching"
    PROBE_THEN_SAE = "probe_then_sae"
    DLA_ONLY = "dla_only"
    CONTRAST_THEN_STEER = "contrast_then_steer"
    CUSTOM = "custom"


class DatasetSpec(StrictBaseModel):
    dataset_id: str
    source: Literal["builtin", "huggingface", "generated", "user_supplied"]
    n_samples: int = Field(gt=0)
    split: Literal["train", "dev", "test", "heldout"] = "dev"
    seed: int = 0
    generator_id: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContrastSpec(StrictBaseModel):
    """Pairs that define the behavioral contrast (e.g. ABBA vs BABA in IOI)."""

    contrast_id: str
    positive_template: str
    negative_template: str
    pairing: Literal["matched", "unmatched"] = "matched"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Budget(StrictBaseModel):
    max_tokens: int | None = None
    max_gpu_seconds: float | None = None
    max_samples: int | None = None
    max_tool_calls: int | None = None
    max_wallclock_seconds: float | None = None


_FORBIDDEN_TOKENS = (
    "import os", "import sys", "import subprocess", "import socket",
    "import shutil", "import requests", "import urllib", "import http",
    "open(", "__import__", "eval(", "exec(", "compile(", "globals(",
    "getattr(", "setattr(", "delattr(", "breakpoint(",
)


class CustomMetricDef(StrictBaseModel):
    """Inline metric definition, frozen at spec finalize.

    The agent cannot edit a custom metric after the spec is approved — the
    runtime checks that ``source_hash`` matches what was finalized. The
    integrity story relies on the human reviewing ``source_code`` at
    ``finalize_spec``; the runtime restriction is defense in depth, not a
    sandbox.
    """

    name: str = Field(min_length=2, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=4)
    family: MetricFamily
    value_range: tuple[float | None, float | None]
    direction: Literal["higher", "lower", "either"]
    requires_inputs: list[str] = Field(min_length=1)
    function_name: str = Field(default="compute", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    source_code: str = Field(min_length=10)
    source_hash: str = ""

    @model_validator(mode="after")
    def _hash_and_compile_check(self) -> "CustomMetricDef":
        import ast
        import hashlib

        if not self.source_hash:
            self.source_hash = hashlib.sha256(self.source_code.encode("utf-8")).hexdigest()
        else:
            actual = hashlib.sha256(self.source_code.encode("utf-8")).hexdigest()
            if actual != self.source_hash:
                raise ValueError(
                    f"custom metric {self.name!r}: source_hash mismatch "
                    f"(expected {self.source_hash[:12]}, got {actual[:12]})"
                )

        try:
            tree = ast.parse(self.source_code)
        except SyntaxError as e:
            raise ValueError(f"custom metric {self.name!r} source does not parse: {e}")
        defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        if self.function_name not in defs:
            raise ValueError(
                f"custom metric {self.name!r} must define def "
                f"{self.function_name}(inputs: dict) -> float"
            )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError(
                    f"custom metric {self.name!r}: import statements are not "
                    f"allowed. The runtime pre-injects `math`; use it directly."
                )
        for forbidden in _FORBIDDEN_TOKENS:
            if forbidden in self.source_code:
                raise ValueError(
                    f"custom metric {self.name!r} contains forbidden token "
                    f"{forbidden!r}; pure-Python math only"
                )
        return self


class Criterion(StrictBaseModel):
    """A pre-registered, falsifiable success criterion.

    Frozen after approval. Editing requires a new spec revision.
    """

    criterion_id: str
    description: str
    metric: MetricName
    comparator: Literal[">=", ">", "<=", "<", "=="]
    threshold: float
    n_min: int | None = None
    on_split: Literal["dev", "test", "heldout"] = "heldout"
    metric_params: dict[str, Any] = Field(default_factory=dict)
    custom_metric_def: CustomMetricDef | None = None

    @model_validator(mode="after")
    def _custom_metric_has_def(self) -> "Criterion":
        if self.metric == MetricName.CUSTOM and self.custom_metric_def is None:
            raise ValueError("metric=CUSTOM requires custom_metric_def")
        if self.metric != MetricName.CUSTOM and self.custom_metric_def is not None:
            raise ValueError("custom_metric_def only allowed when metric=CUSTOM")
        return self

    @model_validator(mode="after")
    def _threshold_in_metric_range(self) -> "Criterion":
        if self.metric == MetricName.CUSTOM and self.custom_metric_def is not None:
            lo, hi = self.custom_metric_def.value_range
            if lo is not None and self.threshold < lo:
                raise ValueError(
                    f"threshold {self.threshold} below custom metric "
                    f"{self.custom_metric_def.name!r} range "
                    f"[{lo}, {hi if hi is not None else 'inf'}]"
                )
            if hi is not None and self.threshold > hi:
                raise ValueError(
                    f"threshold {self.threshold} above custom metric "
                    f"{self.custom_metric_def.name!r} range "
                    f"[{lo if lo is not None else '-inf'}, {hi}]"
                )
            return self
        meta = METRIC_META.get(self.metric)
        if meta is None:
            return self
        lo, hi = meta.value_range
        if lo is not None and self.threshold < lo:
            raise ValueError(
                f"threshold {self.threshold} below {self.metric.value} valid "
                f"range [{lo}, {hi if hi is not None else 'inf'}]"
            )
        if hi is not None and self.threshold > hi:
            raise ValueError(
                f"threshold {self.threshold} above {self.metric.value} valid "
                f"range [{lo if lo is not None else '-inf'}, {hi}]"
            )
        return self

    @model_validator(mode="after")
    def _comparator_matches_metric_direction(self) -> "Criterion":
        if self.metric == MetricName.CUSTOM and self.custom_metric_def is not None:
            direction = self.custom_metric_def.direction
            display = self.custom_metric_def.name
        else:
            meta = METRIC_META.get(self.metric)
            if meta is None:
                return self
            direction = meta.direction
            display = self.metric.value
        if direction == "either":
            return self
        higher = {">=", ">"}
        lower = {"<=", "<"}
        comp_dir = (
            "higher" if self.comparator in higher
            else "lower" if self.comparator in lower
            else None
        )
        if comp_dir is not None and comp_dir != direction:
            raise ValueError(
                f"comparator {self.comparator!r} suggests {comp_dir}-is-better "
                f"but {display} is {direction}-is-better"
            )
        return self


class AbortPredicate(StrictBaseModel):
    """A typed abort condition: stop the run if the named metric crosses a bound.

    Mechanically evaluated by the investigation pipeline after each metric
    commit. Free-text entries in ``abort_if`` are still accepted but are
    advisory only — the pipeline will not act on them.
    """

    predicate_id: str
    description: str
    metric: MetricName
    comparator: Literal[">=", ">", "<=", "<", "=="]
    threshold: float
    on_stage: InvestigationStage | None = None
    metric_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _threshold_in_metric_range(self) -> "AbortPredicate":
        meta = METRIC_META.get(self.metric)
        if meta is None:
            return self
        lo, hi = meta.value_range
        if lo is not None and self.threshold < lo:
            raise ValueError(
                f"threshold {self.threshold} below {self.metric.value} valid "
                f"range [{lo}, {hi if hi is not None else 'inf'}]"
            )
        if hi is not None and self.threshold > hi:
            raise ValueError(
                f"threshold {self.threshold} above {self.metric.value} valid "
                f"range [{lo if lo is not None else '-inf'}, {hi}]"
            )
        return self


class StageSpec(StrictBaseModel):
    stage: InvestigationStage
    pattern: PatternId
    tools: list[ToolName]
    metrics: list[MetricName]
    budget: Budget = Field(default_factory=Budget)
    notes: str | None = None

    @model_validator(mode="after")
    def _has_tools(self) -> "StageSpec":
        if not self.tools:
            raise ValueError(f"stage {self.stage} must declare at least one tool")
        return self


class CritiqueNote(StrictBaseModel):
    note_id: str
    severity: Literal["info", "warn", "block"]
    target_field: str | None = None
    message: str
    proposed_fix: str | None = None


class Approval(StrictBaseModel):
    approver: str
    approver_kind: Literal["human", "agent"]
    approved_at: datetime
    notes: str | None = None


class InvestigationSpec(StrictBaseModel):
    """Pre-registered investigation contract. Stage 1+ refuse to run without one."""

    spec_id: str
    revision: int = Field(ge=1, default=1)
    parent_spec_id: str | None = None
    prior_results_ref: str | None = None
    revision_reason: str | None = None

    question: str
    hypothesis: str
    phenomenon_id: str
    behavior: BehaviorSpec
    model: ModelRef
    dataset: DatasetSpec
    contrast: ContrastSpec | None = None

    stages: list[StageSpec]
    success_criteria: list[Criterion]
    abort_if: list[AbortPredicate | str] = Field(default_factory=list)
    max_revisions: int = Field(ge=1, default=5)

    budget: Budget = Field(default_factory=Budget)
    risks: list[str] = Field(default_factory=list)

    status: SpecStatus = SpecStatus.DRAFT
    critiques: list[CritiqueNote] = Field(default_factory=list)
    approval: Approval | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _revision_has_parent(self) -> "InvestigationSpec":
        if self.revision > 1 and self.parent_spec_id is None:
            raise ValueError("revisions > 1 must reference a parent_spec_id")
        if self.parent_spec_id is not None and self.prior_results_ref is None:
            raise ValueError(
                "a child spec must reference the prior_results_ref that motivated it"
            )
        return self

    @model_validator(mode="after")
    def _has_falsifiable_criteria(self) -> "InvestigationSpec":
        if not self.success_criteria:
            raise ValueError("spec must declare at least one success criterion")
        return self

    @model_validator(mode="after")
    def _approved_specs_have_approval(self) -> "InvestigationSpec":
        if self.status == SpecStatus.APPROVED and self.approval is None:
            raise ValueError("approved specs must carry an Approval record")
        return self

    @model_validator(mode="after")
    def _stages_unique(self) -> "InvestigationSpec":
        seen = [s.stage for s in self.stages]
        if len(seen) != len(set(seen)):
            raise ValueError("stages must be unique by InvestigationStage")
        return self

    @model_validator(mode="after")
    def _tool_requirements_satisfied(self) -> "InvestigationSpec":
        """Every tool used in any stage must have its required spec-level
        fields populated (e.g. patching tools require `contrast`).
        """
        spec_view = {"contrast": self.contrast}
        for stage in self.stages:
            for tool in stage.tools:
                meta = TOOL_META.get(tool)
                if meta is None:
                    continue
                for required in meta.requires_fields:
                    if not spec_view.get(required):
                        raise ValueError(
                            f"tool {tool.value!r} in stage {stage.stage.value!r} "
                            f"requires the spec-level field {required!r} to be set"
                        )
        return self


def typed_abort_predicates(spec: "InvestigationSpec") -> list[AbortPredicate]:
    """Return only the typed (mechanically-evaluable) entries from spec.abort_if.

    Free-text entries are advisory and ignored by the investigation gate.
    """
    return [p for p in spec.abort_if if isinstance(p, AbortPredicate)]


class SpecRevisionLink(StrictBaseModel):
    """Edge in the spec DAG: child spec was produced from parent + results."""

    parent_spec_id: str
    child_spec_id: str
    prior_results_ref: str
    revision_reason: str
    outcome_so_far: InvestigationOutcome = InvestigationOutcome.PENDING
