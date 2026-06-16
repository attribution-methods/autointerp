# effect_size

**Family:** behavioral · **Range:** unbounded (typically -∞..∞) · **Direction:** higher is better

Standardized difference between two distributions (Cohen's d by default).
Reports practical significance, not just statistical significance.

## When to use
- Comparing behavior under two conditions when both are noisy.
- Power calculations during Stage 0 — pick `n` to detect d ≥ 0.5 with α=0.05.

## Pitfalls
- d depends on the pooled SD; report the SD alongside.
- For paired contrast designs prefer Cohen's d_z (paired) over the unpaired form.
