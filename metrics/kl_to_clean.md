# kl_to_clean

**Family:** localization · **Range:** [0, ∞) · **Direction:** lower is better

KL divergence between an intervened-output distribution and the clean (unmodified)
output distribution. Quantifies how much an intervention changes model behavior.

## When to use
- Activation patching / ablations: how distinct is the intervened run from clean?
- Logit-lens style sweeps: KL between intermediate layer's projected distribution
  and final output.

## Pitfalls
- "KL drop ≥ 50%" in prose means the *reduction* in KL relative to a baseline,
  not the metric value itself. Re-express such claims as `patch_effect_recovery`
  or as an explicit ratio criterion.
- KL is asymmetric. State direction (KL(intervened || clean) vs the reverse).
