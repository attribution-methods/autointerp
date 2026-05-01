---
name: activation-patching
description: Activation patching, causal tracing, activation replacement, swaps, ablations, and component localization. Use to test whether a layer, token position, attention head, MLP output, or residual stream site causally transfers a behavior or restores a corrupted computation.
---

# Activation Patching

## Workflow

1. Define clean and corrupted prompts that isolate the behavior or answer.
2. Choose the metric before patching, such as target logit difference, classifier score, or generation outcome.
3. Patch a coarse grid of residual layers and positions.
4. Narrow to MLP, attention, and head components only after a coarse site is promising.
5. Validate the final site with held-out prompt pairs and ablations.

## Tools

Use `../../src/autointerp/tools/patching.py`:

```python
from autointerp.tools.activations import capture_activations
from autointerp.tools.patching import patch_generation

source = capture_activations(handle, [clean_prompt], layer=30, token_index=-1)[0]
patched = patch_generation(handle, corrupted_prompt, source, layer=30, patch_positions=[-1])
```

## Evidence

A site is stronger evidence when it transfers the behavior in both directions, survives prompt paraphrases, and affects a predeclared metric rather than only one cherry-picked sample.

## Cautions

Generation patching can be noisy. Prefer logit-level metrics for sweeps, then inspect generations only for top sites.
