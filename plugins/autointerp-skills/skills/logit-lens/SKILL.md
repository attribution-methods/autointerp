---
name: logit-lens
description: Logit lens analysis for decoding intermediate residual stream predictions. Use when asking what tokens a layer appears to predict, where output-relevant information emerges, which positions have high intermediate-final KL divergence, or whether a hidden behavior appears in latent token predictions.
---

# Logit Lens

## Workflow

1. Choose a prompt or cached activation set with token positions relevant to the behavior.
2. Decode intermediate residual states with the unembedding matrix, optionally after final normalization.
3. Inspect top tokens by layer and position.
4. Filter positions by KL divergence from final logits when searching large prompt sets.
5. Treat decoded tokens as suggestive evidence, then validate with patching, steering, or prompt counterfactuals.

## Tools

Use `../../src/autointerp/tools/lenses.py`:

```python
from autointerp.tools.lenses import logit_lens

rows = logit_lens(handle, prompt, layers=[20, 30, 40, 50], top_k=16)
for row in rows:
    print(row["layer"], row["kl_to_final"], row["top_tokens"][:5])
```

## Interpretation

Look for specific names, topics, goals, refusals, answers, or behavior-relevant words that appear earlier than the final distribution would suggest.

## Cautions

The plain logit lens is not calibrated across layers. Layernorm handling matters. Use tuned lens when calibrated intermediate predictions are central to the claim.
