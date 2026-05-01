---
name: mlp-neuron-analysis
description: MLP, neuron, and feed-forward feature analysis. Use for ranking neurons or MLP outputs by activation, contribution, logit effect, concept selectivity, bilinear interactions, or causal impact on a behavior.
---

# MLP Neuron Analysis

## Workflow

1. Identify candidate layers from patching, probes, or lens emergence.
2. Extract MLP activations and outputs for behavior-present and control prompts.
3. Rank neurons by activation difference, attribution score, or direct logit effect.
4. Inspect top neurons across diverse prompts to avoid prompt-specific artifacts.
5. Validate with neuron ablation, MLP output patching, or feature-level steering.

## Implementation Pattern

Use component specs such as `L30MLP` with `activation-cache`, then use `gradient-attribution` or `circuit-tracing` to rank sites.

```python
mlp_acts = capture_activations(handle, prompts, layer=30, component="mlp", token_index=-1)
```

## Evidence

MLP evidence is strongest when a small set of neurons or features predicts the behavior and interventions change the metric in the expected direction.

## Cautions

Individual neurons are often polysemantic. Prefer feature clusters or subspaces when single-neuron evidence is unstable.
