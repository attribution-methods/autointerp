# sufficiency

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

Behavior reproduced when **only the candidate component** is active and
everything else is ablated. Single-component dual of `completeness`.

`sufficiency = only_component_metric / full_model_metric`

## When to use
- Quick "does this single head/MLP carry the behavior on its own?" check
  before assembling a multi-component circuit.
- Pair with `necessity_drop` for a lightweight 2x2 importance grid:
  necessary + sufficient (load-bearing) vs. either-only (redundant or
  routing).

## Pitfalls
- A high `sufficiency` for a single component does NOT prove monosemanticity
  — the component may also fire for unrelated behaviors. Confirm with
  off-distribution probes.
- Ratio is undefined when `full_model_metric == 0`. The canonical impl
  raises in that case rather than silently returning 0/0.
