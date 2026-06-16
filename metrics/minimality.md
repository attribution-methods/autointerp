# minimality

**Family:** causal · **Range:** [0, ∞) · **Direction:** higher is better

Drop in `faithfulness` when removing any single component from the circuit.
A high minimality score means every component pulls weight; low means the
circuit is bloated.

## When to use
- After identifying a faithful circuit, sweep over components and re-measure
  faithfulness with each one removed.
- Reported as the *worst* (largest) drop, or the distribution of drops.

## Pitfalls
- Combinatorial in circuit size; cap k or use greedy pruning.
- A component with low individual minimality score may still be necessary
  jointly with another — pairwise checks are stronger than singleton checks.
