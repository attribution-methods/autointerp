"""Shared artifact schemas for automated interpretability investigations.

These schemas are intentionally method-neutral. They let black-box probes,
white-box activation work, circuit search, SAE inspection, and downstream
projects exchange evidence without depending on one agent runtime.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base class that forbids silent schema drift in saved artifacts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceStrength(str, Enum):
    UNKNOWN = "unknown"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class InvestigationStage(str, Enum):
    SETUP = "setup"
    BLACK_BOX = "black_box"
    LOCALIZATION = "localization"
    ACTIVATION_ANALYSIS = "activation_analysis"
    FEATURE_ANALYSIS = "feature_analysis"
    INTERVENTION = "intervention"
    VALIDATION = "validation"
    REPORTING = "reporting"


class ModelRef(StrictBaseModel):
    """A model identity that can describe local checkpoints or API targets."""

    model_id: str
    provider: str | None = None
    revision: str | None = None
    dtype: str | None = None
    device: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(StrictBaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class BehaviorSpec(StrictBaseModel):
    """The behavior, failure mode, or latent property under investigation."""

    behavior_id: str
    description: str
    risk_area: str | None = None
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("behavior_id")
    @classmethod
    def _behavior_id_is_stable(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("behavior_id must be non-empty")
        return value


class PromptCase(StrictBaseModel):
    prompt_id: str
    messages: list[ChatMessage]
    prefill: str | None = None
    expected_behavior: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_messages(self) -> "PromptCase":
        if not self.messages:
            raise ValueError("PromptCase must contain at least one message")
        return self


class PromptBatch(StrictBaseModel):
    batch_id: str
    behavior_id: str
    cases: list[PromptCase]
    split: Literal["train", "dev", "test", "heldout", "mixed"] = "mixed"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_prompt_ids(self) -> "PromptBatch":
        ids = [case.prompt_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("PromptBatch prompt_id values must be unique")
        return self


class GenerationSample(StrictBaseModel):
    sample_id: str
    prompt_id: str
    model: ModelRef
    output: str
    prefill: str | None = None
    temperature: float | None = None
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(StrictBaseModel):
    metric_id: str
    value: float
    numerator: float | None = None
    denominator: float | None = None
    threshold: float | None = None
    passed: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BehavioralFinding(StrictBaseModel):
    finding_id: str
    behavior_id: str
    stage: InvestigationStage = InvestigationStage.BLACK_BOX
    summary: str
    evidence_strength: EvidenceStrength = EvidenceStrength.UNKNOWN
    sample_ids: list[str] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActivationCacheRef(StrictBaseModel):
    cache_id: str
    model: ModelRef
    source_path: str
    prompt_batch_id: str
    layers: list[int | str]
    components: list[str] = Field(default_factory=list)
    token_selector: str | None = None
    dtype: str | None = None
    shape: list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateSite(StrictBaseModel):
    """A component, token, feature, or prompt-surface hypothesis to test."""

    site_id: str
    source: str
    method: str
    score: float | None = None
    layer: int | None = None
    component: str | None = None
    head_index: int | None = None
    token_index: int | None = None
    feature_id: int | str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureFinding(StrictBaseModel):
    feature_id: int | str
    source: str
    label: str | None = None
    activation: float | None = None
    layer: int | None = None
    component: str | None = None
    token: str | None = None
    examples: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterventionResult(StrictBaseModel):
    intervention_id: str
    method: str
    candidate_site_id: str | None = None
    prompt_ids: list[str] = Field(default_factory=list)
    baseline_metric: MetricResult | None = None
    intervention_metric: MetricResult | None = None
    delta: float | None = None
    passed: bool | None = None
    output_sample_ids: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(StrictBaseModel):
    validation_id: str
    hypothesis: str
    method: str
    behavior_id: str
    candidate_site_ids: list[str] = Field(default_factory=list)
    intervention_results: list[InterventionResult] = Field(default_factory=list)
    heldout_prompt_batch_id: str | None = None
    passed: bool
    evidence_strength: EvidenceStrength = EvidenceStrength.UNKNOWN
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InvestigationReport(StrictBaseModel):
    report_id: str
    behavior: BehaviorSpec
    models: list[ModelRef] = Field(default_factory=list)
    prompt_batches: list[PromptBatch] = Field(default_factory=list)
    samples: list[GenerationSample] = Field(default_factory=list)
    findings: list[BehavioralFinding] = Field(default_factory=list)
    activation_caches: list[ActivationCacheRef] = Field(default_factory=list)
    candidate_sites: list[CandidateSite] = Field(default_factory=list)
    feature_findings: list[FeatureFinding] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def write_json(self, path: str | Path, indent: int = 2) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=indent) + "\n")

    @classmethod
    def read_json(cls, path: str | Path) -> "InvestigationReport":
        return cls.model_validate_json(Path(path).read_text())

    def as_json_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

