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

