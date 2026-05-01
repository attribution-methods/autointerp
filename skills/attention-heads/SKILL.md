---
name: attention-heads
description: Attention-head analysis, discovery, visualization, ablation, and head-level intervention. Use for finding heads that discriminate conditions, copy or route tokens, attend to behavior-relevant positions, or causally affect a target output.
---

# Attention Heads

## Workflow

1. Collect prompts that contrast behavior-present and behavior-absent conditions.
2. Extract attention patterns and head outputs across a layer range.
3. Rank heads by discriminability, attention to target positions, or output contribution.
4. Inspect top heads with token-level attention maps and OV vocabulary profiles.
5. Validate with head ablation, patching, or QK/OV decomposition.

## Implementation Pattern

Use `activation-cache` for extraction and `qk-ov-decomposition` for deeper head analysis. Store head sites as `L{layer}H{head}`.

```python
spec = "L24H7"
```

## Evidence

A useful head should have both a readable attention pattern and measurable effect on a predeclared metric.

## Cautions

Attention weights alone are not explanations. Always check the value/output path or causal effect.
