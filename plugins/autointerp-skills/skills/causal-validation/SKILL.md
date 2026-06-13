---
name: causal-validation
description: Causal validation for interpretability claims. Use for ablations, activation swaps, direction sign tests, held-out prompt tests, baseline comparisons, dose-response sweeps, random controls, and reviewer-style checks before accepting a mechanism.
---

# Causal Validation

## Workflow

1. Convert the hypothesis into a predeclared metric and expected intervention direction.
2. Test both positive and negative controls.
3. Run held-out prompts that differ from the discovery prompts.
4. Sweep intervention strength or patch site size and look for dose response.
5. Check whether the intervention affects the target behavior more than unrelated behavior.
6. Record failures and revise the mechanism.

## Common Tests

- Ablate the candidate site and expect the behavior to weaken.
- Patch behavior-present activations into a control prompt and expect the behavior to appear.
- Patch control activations into a behavior-present prompt and expect the behavior to weaken.
- Reverse a steering direction and expect an opposite or null effect.
- Use random same-norm directions as controls.

## Tool Links

Use `activation-patching`, `activation-steering`, `gradient-attribution`, and `circuit-tracing` together. Store validation results separately from discovery results.

## Cautions

Do not call a method causal because it is visually compelling, decodable, or correlated. Causal claims require interventions.
