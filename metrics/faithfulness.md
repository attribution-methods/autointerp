# faithfulness

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

Fraction of the full model's behavior reproduced when running through the
candidate circuit alone (rest of the network mean-ablated or zeroed). Wang et
al. (2022) IOI definition.

## When to use
- Final causal validation step. The headline number for circuit claims.
- Always evaluate on a heldout split, not the discovery split.

## Pitfalls
- Sensitive to the ablation choice (zero / mean / resample). State the choice.
- Reporting faithfulness without `completeness` and `minimality` is misleading;
  all three together pin down a circuit claim.
