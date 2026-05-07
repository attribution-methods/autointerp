# mean_ablation_auc_k

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

Trapezoidal area under the *normalized ablation-delta curve* over a top-K
sweep, averaged across contrast pairs. For each K in the K-grid, the
discovery harness ablates the top-K candidate features (or circuit sites)
and measures the normalized behavioral delta on a fixed benchmark; the
curve at one pair is `[delta_K1, delta_K2, …, delta_KN]`, each in `[0, 1]`.

This is the primary reward used in the
[circuitbreaker auto_circuit_discovery](https://github.com/attribution-methods/circuitbreaker/tree/auto-circuit-discovery)
harness and is what the discovery sub-agent hill-climbs.

## When to use
- Discovery-stage reward when you want a *single scalar* that summarizes
  "how quickly does ablating the top features destroy the behavior."
- Comparing multiple ranking algorithms / SAE-feature scoring functions
  on the same benchmark.

## Required inputs
- `delta_curve_per_pair: list[list[float]]` — one curve per (clean,
  corrupted) pair, each curve length ≥ 2, values normalized to `[0, 1]`.
- `k_grid: list[float] | None` — optional explicit K values for non-uniform
  sampling; defaults to natural index `0..N-1`.

## Pitfalls
- Normalization is the metric's responsibility *of the caller*. Pass
  un-normalized deltas and you'll get an `out-of-range` error. The
  convention from the reference harness is `delta_K / max_observed_delta`
  per pair.
- A single very-easy pair can dominate the mean. If pairs are heterogeneous,
  report the per-pair AUC distribution in `metadata`.
- AUC_K is degenerate if every pair's max delta is achieved at K=1; that
  signals you should narrow the K-grid, not that the algorithm is perfect.
