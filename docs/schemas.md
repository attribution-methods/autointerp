# Artifact Schemas

`autointerp.schemas` defines the shared data layer for agent runs and downstream
integrations. The goal is to keep method outputs portable across black-box
audits, activation localization, SAE inspection, circuit search, and validation.

Core objects:

- `BehaviorSpec`: the behavior or failure mode under investigation.
- `PromptCase` and `PromptBatch`: reproducible prompts and held-out splits.
- `GenerationSample`: model outputs with model and sampling metadata.
- `BehavioralFinding`: evidence gathered during discovery.
- `ActivationCacheRef`: a pointer to cached activations without embedding tensors
  in the report JSON.
- `CandidateSite`: a layer, component, feature, token, or prompt-surface
  hypothesis that should be tested.
- `FeatureFinding`: SAE or feature-level evidence.
- `InterventionResult` and `ValidationResult`: causal or held-out checks.
- `InvestigationReport`: the top-level JSON artifact for a run.

`autointerp.spec` adds the Stage 0 specification layer — the pre-registered
contract for an investigation, drafted conversationally before any run:

- `InvestigationSpec`: the top-level pre-registered plan.
- `StageSpec`, `Criterion`, `DatasetSpec`, `ContrastSpec`, `Budget`: nested
  components of the spec.
- `Approval`, `CritiqueNote`, `SpecRevisionLink`: workflow and revision-DAG.
- `MetricName`, `MetricFamily`, `ToolName`, `PatternId`, `SpecStatus`,
  `InvestigationOutcome`: closed enum vocabularies — extend by PR, never at
  runtime.
- `MetricMeta` and `ToolMeta` (with the `METRIC_META` and `TOOL_META`
  registries): typed contracts for each enum value (range, direction,
  required inputs, required spec fields). The pydantic validators on
  `Criterion` and `InvestigationSpec` read these to enforce that thresholds
  are in range, comparators match metric direction, and tools requiring
  spec-level fields (e.g. `activation_patching` requires `contrast`) only
  appear in stages where those fields are populated.

See [stage0.md](stage0.md) for how the spec is drafted, validated, and approved.

Design rules:

- Treat schemas as contracts. Add optional fields instead of breaking existing
  required fields.
- Store large tensors, logits, and activation caches outside the report. Reference
  them through `ActivationCacheRef.source_path` or metadata.
- Keep discovery and validation separate. A `BehavioralFinding` can be noisy; a
  `ValidationResult` should describe the held-out prompts, controls, or causal
  intervention used to test it.
- Use `metadata` for method-specific detail, but promote fields to the schema
  when multiple methods need them.

Round-trip example:

```python
from autointerp.schemas import BehaviorSpec, InvestigationReport

report = InvestigationReport(
    report_id="demo",
    behavior=BehaviorSpec(
        behavior_id="sycophancy",
        description="Model agrees with false user claims.",
    ),
)
report.write_json("outputs/report.json")
loaded = InvestigationReport.read_json("outputs/report.json")
```

