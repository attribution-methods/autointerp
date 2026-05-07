# mean_steering_auc_k

**Family:** feature · **Range:** [0, 1] · **Direction:** higher is better

Trapezoidal area under the *normalized steering-delta curve* over a top-K
sweep, averaged across contrast pairs. Symmetric counterpart of
`mean_ablation_auc_k`: instead of suppressing the top-K features, the
harness *adds* them as a steering vector and measures the corresponding
behavioral delta.

A high `mean_steering_auc_k` says the discovered features are sufficient
to drive the behavior, not just necessary for it.

## When to use
- Discovery-stage reward whenever you want to credit features that
  *induce* the target behavior, not only those whose absence breaks it.
- As one half of `combined_auc_k` (the default discovery reward).

## Required inputs
- `delta_curve_per_pair: list[list[float]]` — see `mean_ablation_auc_k`.
- `k_grid: list[float] | None` — optional explicit K grid.

## Pitfalls
- Steering is *additive* and can take residuals out of distribution; report
  perplexity or any base-rate sanity metric in `metadata` so wild
  off-distribution moves don't masquerade as a great steering reward.
- Steering coefficient choice matters. The reference harness sweeps a fixed
  α; if you change α between runs the AUC values are not comparable.
