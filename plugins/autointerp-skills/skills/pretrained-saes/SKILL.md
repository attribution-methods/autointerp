---
name: pretrained-saes
description: Loading pretrained sparse autoencoders (Gemma Scope v1 / v2, Llama Scope, JumpReLU SAEs) for feature inspection. Use before training your own SAE — pretrained dictionaries cover Gemma 2 / Gemma 3 / Llama-3 / Pythia and come with auto-interp labels via Neuronpedia. The companion sparse-autoencoders skill handles post-processing of the resulting feature activations.
---

# Pretrained SAEs

## When to reach for a pretrained SAE

Almost always, before training your own. Training an SAE on a 7B+ model
is a multi-day GPU job; pretrained dictionaries cover the open-weights
models a typical investigation uses, and they come with two things you
can't easily reproduce:

1. **Auto-interp labels** — every feature has a Neuronpedia-hosted natural
   language description (e.g. `9-RES-JB:14729 → "expressions of joy or
   triumph"`). These let you query features *by semantic neighborhood*
   rather than by feature index.
2. **Max-activating examples** — for each feature, the top tokens it fires
   on across a large corpus. Independent corroboration of the auto-interp
   label.

Train your own SAE only when (a) no pretrained release covers your model
or (b) you need a feature dictionary at a layer / activation stream the
pretrained release doesn't include.

## Available releases

| Release id (sae-lens) | Target model | Notes |
|---|---|---|
| `gemma-scope-2b-pt-res-canonical` | `google/gemma-2-2b` | residual, layer 12, width 16k. |
| `gemma-scope-9b-pt-res-canonical` | `google/gemma-2-9b` | residual, layer 20, width 131k. |
| `gemma-scope-9b-it-res-canonical` | `google/gemma-2-9b-it` | residual, layer 20, width 131k. |
| `llama_scope_lxr_8x` | `meta-llama/Llama-3.1-8B` (base) | residual, all layers, 8x width. Use on Llama-3.1-8B-Instruct too — base→instruct transfer works at slightly higher recon loss. |
| `llama_scope_lxr_32x` | `meta-llama/Llama-3.1-8B` (base) | wider variant of the above. |
| **`goodfire-llama-3.1-8b-instruct`** | `meta-llama/Llama-3.1-8B-Instruct` | trained directly on the instruct model — best fidelity. Single layer (layer_19). |
| `llama-3.1-8b-instruct-andyrdt` | `meta-llama/Llama-3.1-8B-Instruct` | trained on instruct, multiple layers + trainers. `resid_post_layer_{N}_trainer_1`. |
| `llama_scope_r1_distill` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | **NOT vanilla Llama** — only use if targeting the R1-distilled model. |
| JumpReLU SAEs | Pythia (160M / 1.4B / 6.9B / 12B) | DeepMind release. Useful for cross-architecture comparisons. Sparse labels. |

**Targeting Llama-3.1-8B-Instruct?** Prefer `goodfire-llama-3.1-8b-instruct`
or `llama-3.1-8b-instruct-andyrdt` (trained on the instruct model). Fall
back to `llama_scope_lxr_8x` only if you need a layer those don't cover.

Only Gemma Scope (v1 and v2) is fully labeled on Neuronpedia. Llama Scope
labels are partial; Goodfire labels are good for the instruct release;
JumpReLU labels are sparse.

> **Pick your target model first**, then check whether a pretrained SAE
> exists. Don't pick the model based on SAE availability unless the
> investigation specifically needs feature-level granularity — most
> autointerp work can run on residual-stream activations + contrastive
> directions without an SAE at all.

## The standard inspection workflow

```python
from autointerp.tools.sae_loader import load_pretrained_sae
from autointerp.tools.sae_labels import (
    search_features_by_label,
    lookup_neuronpedia_label,
)
from autointerp.tools.sae import top_features, attach_labels

# 1. Load. Lazy import — sae-lens is in the [mechinterp] extra.
sae = load_pretrained_sae(
    release="gemma-scope-9b-it-res",   # see sae-lens registry
    sae_id="layer_20/width_131k/canonical",
)

# 2. Encode activations from your investigation prompts.
#    (autointerp.tools.activations.capture_activations gives you
#    a residual-stream tensor at the right layer.)
feature_acts = sae.encode(activations)        # [batch, seq, n_features]

# 3. Pull top-K features at the decision token.
features = top_features(feature_acts[:, decision_pos, :], top_k=50)

# 4. Attach labels from Neuronpedia (HTTP, cached).
features = attach_labels(
    features,
    labels={f.feature_id: lookup_neuronpedia_label(
        release="gemma-2-9b", layer=20, feature_id=f.feature_id,
    ) for f in features},
)

# 5. Or query by semantic neighborhood instead of dumping all top features.
emotion_features = search_features_by_label(
    release="gemma-2-9b", layer=20,
    query="positive emotion joy triumph satisfaction",
    k=20,
)
```

## Use it for what — three patterns

1. **Feature labeling.** You have a candidate site (e.g. residual stream
   at L20, position=decision token). You want a natural-language
   description of what's there. Encode → top-K → attach labels →
   inspect. Cheap, no causal claim.

2. **Semantic neighborhood projection.** You have a hypothesis
   ("positive emotion fires before behavior X"). You want a 1-D axis on
   the residual stream that picks up that semantic. Search Neuronpedia
   for features matching the description, take their decoder rows
   (`sae.W_dec[feature_ids]`), sum, normalize → a "labeled direction"
   you can project onto. Better than mean-diff of crude prompts.

3. **Causal feature ranking.** You have a behavioral metric (e.g.
   blackmail rate, judge score). For each top SAE feature at the
   decision token, ablate it (`act -= sae.W_dec[i] * sae_act[i]`) and
   re-measure the behavior. Rank features by causal effect. This is the
   transcoder-ranking workflow circuitbreaker uses; for non-transcoder
   SAEs, the same idea applies on whichever stream the SAE was trained
   on.

## Pitfalls

- **Layer choice matters.** Pretrained releases include SAEs at every
  layer; using the wrong one is the most common failure. Decision
  tokens for behavioral commitments tend to live in the upper-mid
  layers (L20–L30 on Gemma 2 9B). Lower layers carry syntax;
  upper layers are nearly the unembed.
- **Width-vs-L0 trade-off.** Gemma Scope ships multiple widths at each
  layer (16k, 65k, 131k, 1M) with different L0 sparsity targets.
  Higher width = more specific features but harder to interpret.
  Start with the `canonical` release (curated by Google) before chasing
  width.
- **Labels are LLM-generated.** Auto-interp labels are themselves
  hypotheses. A label like "expressions of joy" only tells you what an
  explainer model thought when shown 20 max-activating examples. Verify
  with max-activations on YOUR prompts before trusting.
- **Reconstruction loss.** Encoding/decoding through an SAE adds noise.
  When you ablate an SAE feature on the residual stream, you also pay
  the SAE's reconstruction error. For the cleanest causal measurements,
  use the SAE for *site selection* and run the actual ablation on the
  raw residual stream.
- **Gemma 2 vs Gemma 3.** Gemma Scope v1 ≠ Gemma Scope 2. v1 covers
  Gemma 2, v2 covers Gemma 3. They are not interchangeable. Pick the
  one that matches your target model, not the newer one.

## Related skills

- `sparse-autoencoders` — feature-activation post-processing once an SAE
  is loaded (top-K, label filtering, semantic clustering).
- `attention-heads`, `mlp-neuron-analysis` — alternate localization
  primitives when a behavior doesn't decompose cleanly in the SAE basis.
- `causal-validation` — once a feature is labeled and ranked, validate
  causally before publishing.
