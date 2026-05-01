---
name: contrastive-directions
description: Contrastive activation directions and vector geometry. Use for mean-difference vectors, concept directions, refusal or honesty directions, persona vectors, baseline-subtracted concept vectors, cosine comparisons, direction normalization, and vector reuse in steering or causal tests.
---

# Contrastive Directions

## Workflow

1. Define positive and negative prompt sets that differ mainly in the target attribute.
2. Format all prompts identically and extract activations at a fixed layer, component, and token position.
3. Compute the mean positive activation minus the mean negative activation.
4. Normalize only when cosine geometry or steering scale invariance is desired.
5. Validate specificity by testing related concepts, random controls, and held-out prompts.

## Tools

Use `../../src/autointerp/tools/vectors.py`:

```python
from autointerp.tools.vectors import contrastive_direction, cosine_similarity

direction = contrastive_direction(
    handle,
    positive_prompts=honest_transcripts,
    negative_prompts=dishonest_transcripts,
    layer=40,
    token_index=-1,
    normalize=False,
)
```

## Examples

Use honesty vs deception, harmful refusal vs helpful compliance, target persona vs neutral persona, or behavior-present vs behavior-absent transcripts. For single-word concepts, use baseline-subtracted directions with many neutral baseline terms.

## Cautions

Mean-difference vectors can encode prompt style, length, or role markers. Match prompt structure tightly and audit nearest-neighbor concepts before assigning a semantic label.
