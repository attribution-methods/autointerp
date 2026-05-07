---
name: reward-functions
description: Vocabulary and selection guide for reward / score functions used in discovery and intervention stages. Use this when picking what the discovery sub-agent should hill-climb on, or when designing a custom_metric_def for a phenomenon that doesn't fit the canonical AUC family.
---

# Reward Functions

Discovery / hill-climbing stages need a single scalar reward to optimize.
This skill is the vocabulary: what canonical rewards are available, when
each applies, and how to declare a `CustomMetricDef` when none of them fit.

## Canonical rewards (registered in the metric registry)

| Metric | Family | Range | When to use |
|---|---|---|---|
| `auroc` | behavioral | [0, 1] | Binary ranking / separation — does a feature / probe / score rank target above foil? Works for next-token classification, probe outputs, SAE feature activations. |
| `mean_ablation_auc_k` | causal | [0, 1] | Necessity of top-K candidates: ablate them and integrate the normalized behavioral delta over a K-sweep. |
| `mean_steering_auc_k` | feature | [0, 1] | Sufficiency of top-K candidates: steer with them and integrate the same way. |
| `combined_auc_k` | causal | [0, 1] | **Default discovery reward.** 0.5 * (ablation + steering). |
| `patch_effect_recovery` | causal | [0, 1] | Per-component reward when you're patching from clean into corrupted (Wang-et-al style). |
| `logit_diff` | behavioral | ℝ | Quick sanity reward for paired contrasts; not bounded so don't use as a discovery reward without normalization. |
| `kl_to_clean` | localization | [0, ∞) | Lower-is-better; flip the sign or use as an `abort_if`, not a discovery reward. |

Read the per-metric card in `metrics/<name>.md` for input contracts and
pitfalls.

## Choosing a reward

1. **Is the phenomenon binary at one token position (or any per-example
   ranking question)?** → `auroc`.
2. **Is the phenomenon a behavioral contrast (clean vs corrupted)?** →
   `combined_auc_k` (default), or its halves if one mode is irrelevant.
3. **Are you measuring how much patching one component restores
   behavior?** → `patch_effect_recovery`. Less general than AUC family;
   doesn't sweep K.
4. **Does none of the above fit?** → declare a `CustomMetricDef` via
   `propose_custom_metric` at Stage 0.

## Custom rewards

Use `propose_custom_metric` (Stage 0) to define a one-off reward inline.
Constraints worth knowing:
- `requires_inputs` must list every key the function pulls from `inputs`.
- `value_range` must be set; the runtime range-clips and records the
  unclipped value in metadata.
- `direction` chooses whether higher or lower is better (criteria
  comparators must agree).
- `source_code` is hashed at finalize time. The same hash must hold at
  compute time, so the agent cannot tweak the function mid-run.

Example skeleton (custom reward = AUC over heldout, with a confidence
calibration term):

```python
def compute(inputs):
    scores = inputs["scores"]
    labels = inputs["labels"]
    confidence = inputs["confidence"]
    auc = _auc(scores, labels)         # reuse the canonical impl
    ece = _ece(confidence, labels)     # expected calibration error
    return auc - 0.1 * ece
```

## Sanity checks

- A reward that never disagrees with `accuracy` is probably equivalent
  to it; consider whether the more expensive variant adds anything.
- A reward whose top-1 candidate always saturates (≈ 1.0 from iteration 1)
  is too coarse — narrow the K-grid or add a tie-breaker.
- A reward that requires a model forward pass per candidate is fine for
  in-process search but kills an iterative sub-agent. Pre-cache where you
  can.
