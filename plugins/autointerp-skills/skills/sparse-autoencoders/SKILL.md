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

Loading: `autointerp.tools.sae_loader.load_pretrained_sae` for Gemma Scope,
Llama Scope, JumpReLU. See the `pretrained-saes` skill for release picks.

Post-processing once an SAE is loaded:

```python
from autointerp.tools.sae_loader import load_pretrained_sae
from autointerp.tools.sae_labels import lookup_neuronpedia_label, search_features_by_label
from autointerp.tools.sae import top_features, attach_labels, filter_semantic_features

handle = load_pretrained_sae("gemma-scope-9b-pt-res-canonical")
feature_acts = handle.sae.encode(activations)        # [batch, seq, n_features]
features = top_features(feature_acts, top_k=50)
labels = {f.feature_id: lookup_neuronpedia_label(handle.labels_release, handle.layer, f.feature_id) for f in features}
features = attach_labels(features, labels)
semantic = filter_semantic_features(features)
```

## Evidence

Strong SAE evidence names a specific domain, goal, entity, behavior, or representation that predicts useful follow-up tests.

## Cautions

Feature labels are hypotheses. High activation does not prove causal role. Beware generic model-identity and formatting features.
