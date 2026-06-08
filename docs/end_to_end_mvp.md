# End-To-End MVP

The first production target is one reliable investigation path:

1. Define a `BehaviorSpec`.
2. Run cheap black-box probes and controls.
3. Retest on held-out prompt variants.
4. Cache activations only after the behavior is stable enough to justify
   white-box cost.
5. Localize a causal layer or component with ablations or activation patching.
6. Inspect the localized region with logit lens, probes, SAE features, head
   analysis, or circuit tracing.
7. Validate candidate mechanisms with held-out prompts and causal interventions.
8. Save an `InvestigationReport` with claims, limitations, and next steps.

The CI-safe example implements steps 1-3 and writes a schema-valid report:

```bash
python examples/blackbox_to_validation.py
```

The output lands in `outputs/blackbox_to_validation_report.json` by default. It
uses a deterministic local fixture instead of an API or model download so that
CI and contributors can run it anywhere.

For a real local-model run, keep the same report structure and replace the
fixture generator with a function that calls a `ModelHandle`, API target, or
benchmark harness. For white-box follow-up, add:

- an `ActivationCacheRef` after activation capture;
- one or more `CandidateSite` entries after localization;
- `FeatureFinding` rows for SAE/probe/logit-lens evidence;
- `InterventionResult` rows for patching, ablation, or steering checks.

## Productionized path: the investigation pipeline

The example above is a single-script MVP. The productionized version of the
same flow is the **investigation pipeline**
(`src/autointerp/pipelines/investigation/`), which runs an agent through an
approved `InvestigationSpec` under a gated tool surface that enforces
pre-registration:

```bash
python -m autointerp_agent.investigation \
  --spec outputs/specs/<spec_id>_rev<n>.json
```

The pipeline produces the same artifact types — `PromptBatch`,
`ActivationCacheRef`, `CandidateSite`, `FeatureFinding`,
`InterventionResult`, `ValidationResult`, `MetricResult`,
`InvestigationReport` — but each one is committed through `commit_artifact`,
split-tagged, and recorded in a durable run directory at
`runs/<spec_id>_rev<n>/`.

See [investigation.md](investigation.md) for the gate APIs, run-dir layout,
state schema, and how the `compute_metric` → provenance-token →
`commit_artifact("MetricResult", ...)` flow closes the
"agent-makes-up-a-number" hole.

Every run captures a full debugging transcript on disk:
``assistant_turns.jsonl`` (every LLM message), ``tool_invocations.jsonl``
(every tool call), ``tool_invocations/<iter>_<idx>_<tool>_<id>.txt`` (full
untruncated output bodies), ``log.jsonl`` (gated calls only), plus
``INVESTIGATION_LOG.md`` and ``scripts/`` for the agent's own notes and
code. A run can be replayed end-to-end from these alone.

