---
name: gradient-attribution
description: Gradient attribution, saliency, gradient-times-activation, integrated-gradient style analysis, and parameter or data-attribution triage. Use when ranking token positions, layers, examples, or parameters by influence on a target logit, loss, classifier score, or behavior metric.
---

# Gradient Attribution

## Workflow

1. Choose a scalar objective: target token logprob, logit difference, loss, or external score.
2. Capture gradients at a specific layer, component, or embedding site.
3. Compute gradient-times-activation for token saliency, or integrated gradients when a baseline is meaningful.
4. Rank positions or components and compare against simple perturbation tests.
5. Use causal validation for top-ranked sites.

## Tools

Use `../../src/autointerp/tools/attribution.py`:

```python
from autointerp.tools.attribution import next_token_gradient_x_activation

report = next_token_gradient_x_activation(handle, prompt, target_token=" yes", layer=30)
print(report["selected_score"])
```

## Evidence

Gradients answer local sensitivity questions. They are especially useful for finding positions that affect a selected next token or classifier objective.

## Cautions

Gradients are local, baseline-sensitive, and can be misleading under saturation. Compare with ablation or patching before making behavioral claims.
