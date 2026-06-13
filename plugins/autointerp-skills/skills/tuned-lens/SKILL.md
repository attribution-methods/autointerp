---
name: tuned-lens
description: Tuned lens and learned translator analysis for calibrated intermediate predictions. Use when plain logit lens outputs are noisy, when comparing prediction emergence across layers, or when training small translators from hidden states to final-token distributions.
---

# Tuned Lens

## Workflow

1. Use plain logit lens first to decide whether intermediate decoding is worth calibrating.
2. Collect hidden states and final logits on a representative text corpus.
3. Train one translator per layer or component to map hidden states into the final unembedding space.
4. Evaluate translator quality on held-out prompts before using its predictions as evidence.
5. Compare tuned predictions across layers, tokens, and conditions.

## Implementation Pattern

The core tool layer exposes unembedding helpers in `../../src/autointerp/tools/lenses.py`. A tuned lens adds a learned `torch.nn.Module` before unembedding:

```python
translated = translator(hidden_state)
logits = unembed(handle, translated, apply_final_norm=False)
```

## Evidence

Use tuned lens to answer emergence questions: when a target answer becomes decodable, whether a concept is present before a behavioral response, or which component makes a prediction linearly available.

## Cautions

Do not train and evaluate on the same prompts. A translator can learn corpus shortcuts. Report held-out cross-entropy or KL against final logits.
