---
name: attribution-patching
description: Attribution patching for fast first-order estimates of causal patching effects. Use to triage many layers, heads, MLPs, or token positions before running expensive activation patching sweeps.
---

# Attribution Patching

## Workflow

1. Define clean and corrupted runs plus a scalar metric.
2. Run one backward pass to get gradients of the metric with respect to activations.
3. Estimate each site's patch effect as activation delta dot gradient.
4. Rank sites by absolute estimated effect.
5. Confirm top sites with real activation patching.

## Tools

Use `../../src/autointerp/tools/attribution.py`:

```python
from autointerp.tools.attribution import attribution_patch_score, rank_attributions

score = attribution_patch_score(clean_activation, corrupted_activation, clean_gradient)
ranked = rank_attributions({"L20 resid": score})
```

## Evidence

Attribution patching is a screening method. It is useful when many candidate sites exist and full patching would be too slow.

## Cautions

First-order estimates fail when the computation is nonlinear, saturated, or distribution-shifted by the patch. Never stop at attribution patching for a causal claim.
