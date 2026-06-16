# mean_steering_auc_k

**Family:** feature · **Range:** [0, 1] · **Direction:** higher is better

Area under the normalized steering-delta curve over a top-K sweep, averaged
across contrast pairs. For each pair, steer along the candidate ranking's top-K
sites at several K values, normalize the behavioral delta to [0, 1], integrate
via the trapezoid rule, and normalize by the K-range. A high value means
steering with a *few* of the candidate's top sites already moves the behavior.

## Inputs
- `delta_curve_per_pair`: `list[list[float]]` — one normalized delta curve per
  contrast pair (length >= 2, values in [0, 1]).
- `k_grid` (optional): `list[float]` — K values per column. Defaults to uniform
  index spacing.

## When to use
- Reward for `discover_features`: rewards rankings whose top sites are
  *steerable* (causally sufficient to push behavior), complementing the
  ablation (necessity) view.

## Pitfalls
- Steering effects can saturate or reverse at large magnitudes; fix the steering
  scale across K so the curve reflects site choice, not magnitude.
- Normalize deltas to [0, 1] before calling.
