---
name: sparse-autoencoders
description: Sparse autoencoder feature analysis and labeling. Use for decomposing activations into interpretable features, finding high-activation features at tokens, filtering semantic feature clusters, linking features to behaviors, and selecting features for steering or causal tests.
---

# Sparse Autoencoders

## Workflow

1. Pick a layer and activation stream that has a compatible SAE.
2. Encode cached activations and rank top features by activation magnitude.
3. Attach feature labels or explanations from the SAE provider or local label store.
4. Filter low-level syntax, formatting, and generic assistant features.
5. Group features by token position and inspect coherent semantic clusters.
6. Validate candidate features with ablation, steering, or prompt counterfactuals.

## Tools

Use `../../src/autointerp/tools/sae.py` for post-processing SAE outputs:

```python
from autointerp.tools.sae import top_features, attach_labels, filter_semantic_features

features = top_features(feature_acts, top_k=50)
features = attach_labels(features, label_map)
semantic = filter_semantic_features(features)
```

## Evidence

Strong SAE evidence names a specific domain, goal, entity, behavior, or representation that predicts useful follow-up tests.

## Cautions

Feature labels are hypotheses. High activation does not prove causal role. Beware generic model-identity and formatting features.
