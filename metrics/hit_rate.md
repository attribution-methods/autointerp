# hit_rate

**Family:** behavioral · **Range:** [0, 1] · **Direction:** higher is better

Fraction of generations satisfying a behavioral predicate (e.g. "contains the
target keyword", "passes the auditor judge"). Black-box scoring primitive.

## When to use
- Auditing/red-teaming where outputs are open-ended text.
- Cheap baseline before committing to white-box analysis.

## Pitfalls
- Threshold and predicate must be defined before running, not tuned post-hoc.
- For "≥ N items satisfy condition" claims, encode as a count metric or a
  ratio — do not put N as a `hit_rate` threshold (which is in [0,1]).
