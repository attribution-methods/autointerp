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

### Coarse (residual / whole-layer) — `autointerp.tools.patching`

```python
from autointerp.tools.activations import capture_activations
from autointerp.tools.patching import patch_generation

source = capture_activations(handle, [clean_prompt], layer=30, token_index=-1)[0]
patched = patch_generation(handle, corrupted_prompt, source, layer=30, patch_positions=[-1])
```

### Head-level — `autointerp.tools.head_patching`

The helpers hook the attention output projection (`c_proj` on GPT-2,
`o_proj` on Llama/Qwen/Gemma/Mistral), whose input is the concatenation of
per-head outputs ``z``.

```python
from autointerp.tools.head_patching import (
    cache_head_z, mean_head_z, run_with_head_patches, HeadPatch,
    mean_ablate_heads, head_patch_sweep, path_patch, logit_diff,
)

# 1. Cache per-head z on clean and corrupt batches.
clean_cache = cache_head_z(handle, clean_prompts)        # {layer: [B, S, H, d_head]}
corrupt_cache = cache_head_z(handle, corrupt_prompts)
mean_z = mean_head_z(clean_cache)                        # {layer: [S, H, d_head]}

# 2. Sweep (layer, head) patches: clean z -> corrupt run, measure a metric
#    at the final position. `metric` takes [B, V] logits and returns [B].
target_ids, contrast_ids = ...  # per-prompt token ids you decided to score
metric = lambda logits: logit_diff(logits, target_ids, contrast_ids)
sweep = head_patch_sweep(
    handle, clean_prompts, corrupt_prompts, metric,
    patch_positions=[-1], clean_cache=clean_cache,
)
# sweep["recovery"] is a [L, H] tensor of fractional gap recovery.

# 3. Mean-ablate sites identified empirically by the sweep.
top_sites = [...]  # populate from your own sweep results, not from priors
ablated_logits = mean_ablate_heads(handle, clean_prompts, top_sites, mean_z)

# 4. Path-patch: isolate a sender head's *direct* effect on the logits
#    by holding it to clean while freezing every other head to corrupt.
result = path_patch(handle, clean_prompts, corrupt_prompts,
                    sender=top_sites[0], metric=metric, patch_positions=[-1],
                    clean_cache=clean_cache, corrupt_cache=corrupt_cache)
```

### Circuit faithfulness (complement ablation)

Faithfulness is a different experiment from recovery. Recovery patches circuit
heads INTO a corrupted run. Faithfulness mean-ablates every head NOT in the
circuit (the complement) from a clean run — testing whether the circuit alone
sustains the behavior.

```python
# Given: circuit_heads = [(8,6), (8,10), (9,9), ...] from your localization
# Build the complement: every (layer, head) NOT in the circuit.
n_layers = handle.config.num_hidden_layers
n_heads = handle.config.num_attention_heads
all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
complement = [site for site in all_heads if site not in circuit_heads]

# Mean-ablate the complement on clean prompts.
# mean_z should be computed from the same distribution (clean prompts).
circuit_only_logits = mean_ablate_heads(handle, clean_prompts, complement, mean_z)

# Measure: how much logit_diff survives with only the circuit active?
circuit_metric = logit_diff(circuit_only_logits, target_ids, contrast_ids).mean().item()
full_metric = logit_diff(clean_logits, target_ids, contrast_ids).mean().item()
corrupt_metric = logit_diff(corrupt_logits, target_ids, contrast_ids).mean().item()

# faithfulness = (circuit_metric - corrupt_metric) / (full_metric - corrupt_metric)
```

### When to use which granularity

- **Whole-layer / residual** patching is fast and useful for a coarse first
  pass, but it cannot localize to individual attention heads. If a layer-level
  sweep looks flat (most layers recovering similar fractions of the gap),
  that usually means you need head granularity — not that the computation is
  irreducibly distributed.
- **Head patching** (`head_patch_sweep`) localizes effects to individual
  ``(layer, head)`` sites.
- **Path patching** is required to confirm that an upstream head's effect on
  the output flows *through* a specific downstream head rather than via the
  residual stream broadly.

## Evidence

A site is stronger evidence when it transfers the behavior in both directions, survives prompt paraphrases, and affects a predeclared metric rather than only one cherry-picked sample.

## Cautions

- Generation patching can be noisy. Prefer logit-level metrics for sweeps, then inspect generations only for top sites.
- Ensure clean and corrupt prompts tokenize to the same length so position indices align across runs. Assert this in code before sweeping; a one-token drift silently misaligns all per-position interventions.
- Mean ablation, not zero ablation, is the standard ablation baseline; zero-ablating ``z`` removes a typically nonzero head bias and can give misleading drops.
