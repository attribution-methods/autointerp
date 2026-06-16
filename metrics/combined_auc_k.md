# combined_auc_k

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

`0.5 * (mean_ablation_auc_k + mean_steering_auc_k)`. The default reward for the
`discover_features` hill-climbing sub-agent: it balances necessity (ablation)
and sufficiency (steering) so a candidate ranking is rewarded only when its top
sites are both load-bearing when removed and effective when steered.

## Inputs
Either pre-computed scalars:
- `mean_ablation_auc_k`: `float` in [0, 1]
- `mean_steering_auc_k`: `float` in [0, 1]

or raw per-pair curves:
- `ablation_delta_curve_per_pair`: `list[list[float]]`
- `steering_delta_curve_per_pair`: `list[list[float]]`
- `k_grid` (optional): `list[float]`

## When to use
- Default optimization target for `discover_features`. Pick `mean_ablation_auc_k`
  or `mean_steering_auc_k` instead when only one side is meaningful for the
  behavior.

## Pitfalls
- The 0.5/0.5 weighting is a default, not a law — if your behavior is
  steering-dominated (or ablation-dominated), justify the single-sided metric in
  the spec instead.
- A high discovery-time reward on `dev` is not evidence — re-check on a
  `heldout` split via `compute_metric` before committing a `MetricResult`.
