---
name: qk-ov-decomposition
description: QK and OV decomposition for attention circuits. Use to analyze what a head attends to, which source-destination pairs it prefers, what vocabulary directions its OV path writes, and how attention-head behavior contributes to logits or downstream activations.
---

# QK OV Decomposition

## Workflow

1. Select candidate heads from attention-head ranking or patching.
2. Analyze QK behavior: source/destination token pairs, attention specificity, and condition differences.
3. Analyze OV behavior: vocabulary profile, output directions, and direct logit contribution.
4. Test whether QK routing and OV writing jointly explain the behavior.
5. Validate with targeted attention or OV interventions.

## Implementation Pattern

Use `activation-cache` for head outputs and `../../src/autointerp/tools/lenses.py` for direct logit attribution of output vectors.

```python
from autointerp.tools.lenses import direct_logit_attribution
scores = direct_logit_attribution(handle, {"L20H3": ov_output}, target_token=" yes")
```

## Evidence

Good QK evidence says where information is read from. Good OV evidence says what is written. A circuit claim needs both.

## Cautions

OV vocabulary profiles can overemphasize single-token artifacts. Check multi-token behaviors and downstream effects.
