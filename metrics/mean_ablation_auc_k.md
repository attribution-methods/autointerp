# mean_ablation_auc_k

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

Area under the normalized ablation-delta curve over a top-K sweep, averaged
across contrast pairs. For each pair, ablate the candidate ranking's top-K sites
at several K values, normalize the behavioral delta to [0, 1], integrate the
curve via the trapezoid rule, and normalize by the K-range. A high value means a
*few* of the candidate's top-ranked sites already account for most of the
behavior when removed.

## Inputs
- `delta_curve_per_pair`: `list[list[float]]` — one normalized delta curve per
  contrast pair (length >= 2, values in [0, 1]).
- `k_grid` (optional): `list[float]` — K values per column. Defaults to uniform
  index spacing.

## When to use
- Reward for `discover_features`: rewards rankings whose top sites are causally
  load-bearing under ablation.

## Pitfalls
- Deltas MUST be normalized to [0, 1] before calling — the metric rejects
  out-of-range values rather than silently clipping.
- AUC rewards front-loaded importance; it does not certify a minimal circuit.
  Pair with `minimality` / `faithfulness` on held-out prompts before claiming a
  circuit.
