# auroc

**Family:** behavioral · **Range:** [0, 1] · **Direction:** higher is better

Area under the ROC curve for a binary readout (typically a probe). 0.5 = chance,
1.0 = perfect separation.

## When to use
- Linear/MLP probe quality on cached activations.
- Black-box auditor scoring of "behavior present / absent" generations.

## Pitfalls
- AUROC is insensitive to class imbalance — pair with calibration if you care
  about absolute confidence.
- High AUROC on a small dataset with high-dim features (e.g. 768d on 50 samples)
  can be a leakage / overfitting artifact. Cross-validate.
