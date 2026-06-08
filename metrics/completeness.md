# completeness

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

Behavior reproduced when the **complement** of the circuit is ablated (i.e.
only the circuit is intact). The dual of `faithfulness`.

## When to use
- Always alongside `faithfulness`. A circuit can be faithful but incomplete
  (it produces the behavior, but ablating it doesn't kill the behavior — the
  rest of the network compensates).

## Pitfalls
- Like faithfulness, depends on the ablation distribution. Match the choice
  used for the faithfulness measurement.
