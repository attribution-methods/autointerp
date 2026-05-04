# necessity_drop

**Family:** causal · **Range:** unbounded · **Direction:** higher is better

Drop in a behavioral metric when a component is removed from the **full**
model. Unlike `minimality` (which is a drop within an already-isolated circuit),
`necessity_drop` measures whether a component is essential in vivo.

## When to use
- Ranking candidate components by causal importance before circuit assembly.
- Quick sanity check that a high-DLA component actually matters (DLA is
  correlational, this is causal).

## Pitfalls
- Single-component ablations miss redundancy. A redundant component will show
  a small `necessity_drop` even if it computes the right thing.
