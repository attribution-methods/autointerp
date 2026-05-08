# auroc

**Family:** behavioral · **Range:** [0, 1] · **Direction:** higher is better

ROC AUC for a binary readout. Each example carries a model score (any
monotone proxy for P(target=1) — `logit(target) - logit(foil)`, a probe
output, an SAE feature activation, …) and a binary label (1 = positive /
target, 0 = negative / foil). Computed via Mann-Whitney U with mid-rank
tie handling, so no scipy or torch dependency at metric time. 0.5 = chance,
1.0 = perfect separation, 0.0 = perfect anti-separation.

## When to use
- Linear / MLP probe quality on cached activations.
- Black-box auditor scoring of "behavior present / absent" generations.
- Discovery-stage reward when the question is "does this feature /
  circuit / ranking separate target from foil?" — pair with
  `mean_ablation_auc_k` / `mean_steering_auc_k` if you want a *causal*
  reward instead of a ranking one.

## Required inputs
- `scores: list[float]` — one score per example.
- `labels: list[int]` — `1` for positive, `0` for foil. Must contain
  both classes; equal length to `scores`.

## Pitfalls
- AUROC is insensitive to class imbalance — pair with calibration if you
  care about absolute confidence.
- High AUROC on a small dataset with high-dim features (e.g. 768d on 50
  samples) can be a leakage / overfitting artifact. Cross-validate.
- AUROC is a *ranking* metric. A model that ranks correctly but is poorly
  calibrated still scores 1.0; pair with `logit_diff` if you care about
  absolute margins.
- Don't reuse the same examples for ranker selection and AUROC measurement —
  split-tag your `MetricResult` with `split=heldout` for the criterion.
