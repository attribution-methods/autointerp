"""A small black-box discovery to validation pipeline.

This pipeline is deliberately lightweight: it works with any callable that can
turn a PromptCase into text. Real agents can plug in API models, local models,
or benchmark harnesses while keeping the same artifact schemas.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from autointerp.schemas import (
    BehavioralFinding,
    BehaviorSpec,
    CandidateSite,
    EvidenceStrength,
    GenerationSample,
    InterventionResult,
    InvestigationReport,
    InvestigationStage,
    MetricResult,
    ModelRef,
    PromptBatch,
    PromptCase,
    ValidationResult,
)

PromptGenerator = Callable[[PromptCase], str]


def run_prompt_batch(
    batch: PromptBatch,
    model: ModelRef,
    generator: PromptGenerator,
    temperature: float | None = None,
) -> list[GenerationSample]:
    """Run a prompt batch through a caller-provided generator."""

    samples: list[GenerationSample] = []
    for case in batch.cases:
        output = generator(case)
        samples.append(
            GenerationSample(
                sample_id=f"{batch.batch_id}:{case.prompt_id}",
                prompt_id=case.prompt_id,
                model=model,
                output=output,
                prefill=case.prefill,
                temperature=temperature,
                metadata={"batch_id": batch.batch_id, "split": batch.split},
            )
        )
    return samples


def keyword_hit_rate(
    samples: Sequence[GenerationSample],
    positive_terms: Sequence[str],
    negative_terms: Sequence[str] = (),
    metric_id: str = "keyword_hit_rate",
    threshold: float | None = None,
) -> MetricResult:
    """Score how often outputs contain wanted terms and avoid banned terms."""

    if not samples:
        return MetricResult(metric_id=metric_id, value=0.0, numerator=0.0, denominator=0.0)
    hits = sum(
        _contains_any(sample.output, positive_terms)
        and not _contains_any(sample.output, negative_terms)
        for sample in samples
    )
    value = hits / len(samples)
    return MetricResult(
        metric_id=metric_id,
        value=value,
        numerator=float(hits),
        denominator=float(len(samples)),
        threshold=threshold,
        passed=None if threshold is None else value >= threshold,
        metadata={
            "positive_terms": list(positive_terms),
            "negative_terms": list(negative_terms),
        },
    )


def run_blackbox_validation(
    behavior: BehaviorSpec,
    model: ModelRef,
    discovery_batch: PromptBatch,
    heldout_batch: PromptBatch,
    generator: PromptGenerator,
    positive_terms: Sequence[str],
    negative_terms: Sequence[str] = (),
    threshold: float = 0.5,
    report_id: str | None = None,
) -> InvestigationReport:
    """Run discovery prompts, retest on held-out prompts, and build a report."""

    discovery_samples = run_prompt_batch(discovery_batch, model, generator)
    heldout_samples = run_prompt_batch(heldout_batch, model, generator)
    discovery_metric = keyword_hit_rate(
        discovery_samples,
        positive_terms=positive_terms,
        negative_terms=negative_terms,
        metric_id="discovery_keyword_hit_rate",
        threshold=threshold,
    )
    heldout_metric = keyword_hit_rate(
        heldout_samples,
        positive_terms=positive_terms,
        negative_terms=negative_terms,
        metric_id="heldout_keyword_hit_rate",
        threshold=threshold,
    )
    finding = BehavioralFinding(
        finding_id=f"{behavior.behavior_id}:blackbox-finding",
        behavior_id=behavior.behavior_id,
        stage=InvestigationStage.BLACK_BOX,
        summary=_finding_summary(behavior, discovery_metric),
        evidence_strength=_strength_from_metric(discovery_metric.value),
        sample_ids=[sample.sample_id for sample in discovery_samples],
        metrics=[discovery_metric],
        limitations=[
            "Keyword scoring is a triage metric, not a semantic judge.",
            "Black-box evidence can motivate but cannot prove a mechanism.",
        ],
    )
    candidate = CandidateSite(
        site_id=f"{behavior.behavior_id}:prompt-surface",
        source="black_box",
        method="keyword_hit_rate",
        score=discovery_metric.value,
        evidence_refs=[finding.finding_id],
        metadata={
            "interpretation": "Behavior is visible at the prompt-output surface.",
            "next_whitebox_step": "localize layers with ablations or activation patching",
        },
    )
    intervention = InterventionResult(
        intervention_id=f"{behavior.behavior_id}:heldout-retest",
        method="heldout_prompt_retest",
        candidate_site_id=candidate.site_id,
        prompt_ids=[sample.prompt_id for sample in heldout_samples],
        baseline_metric=discovery_metric,
        intervention_metric=heldout_metric,
        delta=heldout_metric.value - discovery_metric.value,
        passed=heldout_metric.passed,
        output_sample_ids=[sample.sample_id for sample in heldout_samples],
        controls=["heldout prompts"],
    )
    validation = ValidationResult(
        validation_id=f"{behavior.behavior_id}:validation",
        hypothesis=(
            "The behavior appears across held-out prompt variants, not only in "
            "the discovery prompts."
        ),
        method="black_box_heldout_retest",
        behavior_id=behavior.behavior_id,
        candidate_site_ids=[candidate.site_id],
        intervention_results=[intervention],
        heldout_prompt_batch_id=heldout_batch.batch_id,
        passed=bool(heldout_metric.passed),
        evidence_strength=_strength_from_metric(heldout_metric.value),
        limitations=[
            "Held-out prompt validation checks behavioral stability only.",
            "A mechanistic claim still requires activation-level localization.",
        ],
    )
    return InvestigationReport(
        report_id=report_id or f"{behavior.behavior_id}:blackbox-validation",
        behavior=behavior,
        models=[model],
        prompt_batches=[discovery_batch, heldout_batch],
        samples=[*discovery_samples, *heldout_samples],
        findings=[finding],
        candidate_sites=[candidate],
        validations=[validation],
        claims=_claims_for_validation(behavior, validation),
        limitations=[
            "This report is an end-to-end scaffold artifact, not a full causal explanation.",
            "Use activation caching and causal interventions before accepting a mechanism.",
        ],
        next_steps=[
            "Cache activations on the discovery and held-out prompt batches.",
            "Run layer localization with ablations or activation patching.",
            "Inspect only the localized layer range with SAE, probe, or logit-lens tools.",
        ],
    )


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _strength_from_metric(value: float) -> EvidenceStrength:
    if value >= 0.9:
        return EvidenceStrength.STRONG
    if value >= 0.6:
        return EvidenceStrength.MODERATE
    if value > 0.0:
        return EvidenceStrength.WEAK
    return EvidenceStrength.UNKNOWN


def _finding_summary(behavior: BehaviorSpec, metric: MetricResult) -> str:
    denominator = int(metric.denominator or 0)
    numerator = int(metric.numerator or 0)
    return (
        f"{behavior.behavior_id} appeared in {numerator}/{denominator} "
        f"discovery outputs under {metric.metric_id}."
    )


def _claims_for_validation(
    behavior: BehaviorSpec,
    validation: ValidationResult,
) -> list[str]:
    if validation.passed:
        return [
            (
                f"{behavior.behavior_id} reproduced on held-out black-box prompts; "
                "it is ready for activation-level localization."
            )
        ]
    return [
        (
            f"{behavior.behavior_id} did not reproduce strongly enough on held-out "
            "black-box prompts; refine the behavior spec before white-box work."
        )
    ]

