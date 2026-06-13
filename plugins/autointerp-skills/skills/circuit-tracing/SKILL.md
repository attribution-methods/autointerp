---
name: circuit-tracing
description: Circuit tracing, feature graph construction, formula scoring, and circuit diagram synthesis. Use to combine evidence from probes, patching, attribution, logit lens, SAE features, MLPs, and attention heads into a candidate mechanism.
---

# Circuit Tracing

## Workflow

1. Define the behavioral metric and prompt family.
2. Rank candidate sites with cheap methods: lens emergence, probes, gradients, SAE features, and attribution patching.
3. Build a small candidate graph of token positions, layers, heads, MLPs, and features.
4. Run causal validation on edges and nodes, not just isolated sites.
5. Summarize the circuit as claims with confidence levels and failed alternatives.

## Tools

Use `../../src/autointerp/tools/circuits.py`:

```python
from autointerp.tools.circuits import CircuitSite, rank_sites, combine_scores

sites = [CircuitSite("answer-copy-head", 18, "head", 2.4, "patching transfers target token")]
top = rank_sites(sites, top_k=5)
```

## Evidence

A circuit should explain how information is represented, routed, transformed, and read out. A ranked list of sites is not yet a circuit.

## Cautions

Avoid adding every correlated site to the mechanism. Prefer the smallest graph that survives ablations and counterfactual prompts.
