# combined_auc_k

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

`0.5 * (mean_ablation_auc_k + mean_steering_auc_k)`. The default optimization
target for the circuit-discovery sub-agent. Combines necessity (ablation
breaks behavior) and sufficiency (steering induces behavior) into one
hill-climbable scalar.

## When to use
- Default reward for `InvestigationStage.FEATURE_DISCOVERY`.
- Whenever you want a circuit-quality score that is robust to one of the
  two modes giving a degenerate signal (e.g. a feature set that is highly
  necessary but trivially un-steerable, or vice versa).

## Required inputs (either form)

**Pre-aggregated form** (preferred — what the harness emits):
- `mean_ablation_auc_k: float`
- `mean_steering_auc_k: float`

**Raw-curve form**:
- `ablation_delta_curve_per_pair: list[list[float]]`
- `steering_delta_curve_per_pair: list[list[float]]`
- `k_grid: list[float] | None` — shared K-grid for both curves.

## Pitfalls
- Equal weighting is a choice, not a fact. If your phenomenon clearly
  favors one mode (e.g. for refusal circuits, ablation is the natural
  signal) consider declaring a `custom_metric_def` with the weighting you
  want — the AUC primitives are still reusable inside it.
- Don't compute on training-split data. The discovery sub-agent fits to
  whatever signal it sees; reserve heldout pairs for the criterion check.
