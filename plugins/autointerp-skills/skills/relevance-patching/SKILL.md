---
name: relevance-patching
description: Relevance Patching workflows for RelP/ReIP, an LRP-based replacement for gradient signals in attribution patching. Use when ranking residual stream sites, MLP outputs, attention outputs, token positions, or field-level inputs by causal relevance, especially when standard gradient attribution is noisy and model-specific LayerNorm, RMSNorm, activation, gated-MLP, rotary-attention, Qwen, Gemma, GPT-2, or Pythia implementation details matter.
---

# Relevance Patching

Relevance Patching is usually abbreviated `RelP`; issue discussions sometimes
spell it `ReIP`. Treat these as the same method unless the referenced code says
otherwise.

RelP keeps the shape of attribution patching:

1. Run an original input and a patch/corrupt input.
2. Choose a scalar metric, such as a logit difference or classifier score.
3. Score a component with activation delta times a backward signal.

The change is the backward signal. Standard attribution patching uses local
gradients. RelP replaces those gradients with Layer-wise Relevance Propagation
(LRP) coefficients produced by model-specific backward rules.

Use RelP as a fast localization and circuit-discovery method. It is stronger
evidence than raw gradient attribution when implemented correctly, but it is
still an approximation to activation patching.

## Core workflow

1. Define matched original and patch inputs before inspecting scores.
2. Pre-register the scalar metric. Prefer logit differences or a frozen
   classifier score over free-form generation.
3. Select the component granularity: residual stream, MLP output, attention
   output, per-head `z`, SAE feature, or input field.
4. Enable RelP rules for the model family and architecture.
5. Run the original input with activations retained and call backward on the
   metric under the RelP rules.
6. Score each component as the activation difference between patch and original
   runs dotted with the RelP backward coefficient at that component.
7. Rank by signed score for direction-specific hypotheses and by absolute
   score for triage.
8. Confirm top sites with activation patching, ablation, path patching, or
   held-out examples.

## Rule selection

Use these transformer LRP rules:

- `LN-rule`: LayerNorm and RMSNorm. Detach the normalization scale before
  dividing by it. This avoids relevance collapse through normalization.
- `Identity-rule`: GELU, SiLU, and related activation functions. Forward uses
  the true activation; backward behaves like identity via
  `stabilize(x) * (act(x) / stabilize(x)).detach()`.
- `Half-rule`: multiplicative gates. For gated MLP products such as
  `act(gate_proj(x)) * up_proj(x)`, replace the product with
  `(product / 2) + (product / 2).detach()` so relevance is split instead of
  doubled.
- `AH-rule`: attention. Detach the post-softmax attention probabilities before
  multiplying by values. Use this only when the implementation exposes eager
  attention; the RelP paper's main experiments did not use AH-rule, while
  Pando's HuggingFace agent enables it by default.
- Linear layers: use the ordinary linear backward path. The LRP z-rule is
  equivalent to gradient-times-input for standard linear maps.

Default rule sets:

- Paper-style component localization: `LN-rule`, `Identity-rule`, and
  `Half-rule` only when the model has multiplicative gates.
- Pando/HuggingFace-style field attribution: `LN`, `Identity`, `Half`, `AH`,
  with the model loaded using eager attention.

## Model-specific implementation

### GPT-2 family

Architecture facts:

- Normalization: LayerNorm.
- MLP: ungated `GPT2MLP`, roughly `c_fc -> act -> c_proj -> dropout`.
- Attention: standard absolute positional embeddings; GPT-2 attention has a
  GPT-2-specific eager attention signature.
- Multiplicative gates: absent.

Implementation checklist:

- Apply `LN-rule` to every `nn.LayerNorm`.
- Apply `Identity-rule` by wrapping `GPT2MLP.act`, not by replacing `c_fc` or
  `c_proj`.
- Do not apply `Half-rule`.
- If applying `AH-rule` in HuggingFace, patch the GPT-2-style attention
  function separately from Llama/Qwen/Gemma-style attention. Its function
  signature uses `head_mask` and `scale_attn_weights` rather than `scaling`.
- Treat HuggingFace `Conv1D` projections as linear passthrough modules.

TransformerLens-style setup:

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
model.cfg.use_lrp = True
model.cfg.LRP_rules = ["LN-rule", "Identity-rule"]
```

### Pythia / GPT-NeoX family

Architecture facts:

- Normalization: RMSNorm in common Pythia/GPT-NeoX checkpoints.
- MLP: ungated activation MLP.
- Attention: rotary positional embeddings.
- Multiplicative gates: absent.

Implementation checklist:

- Apply `LN-rule` to RMSNorm, not only to `nn.LayerNorm`.
- Apply `Identity-rule` to the MLP activation.
- Do not apply `Half-rule`.
- For attention-level RelP on rotary models, score post-RoPE attention
  probabilities, attention output, or per-head `z`; do not compare raw
  pre-rotary `q`/`k` vectors across positions.
- Prefer residual stream and MLP-output RelP first; these are where the RelP
  paper reported the clearest gains over attribution patching.

### Qwen / Qwen2 / Qwen2.5 family

Architecture facts:

- Normalization: RMSNorm.
- MLP: gated SiLU-style MLP with `gate_proj`, `up_proj`, `down_proj`, and
  `act_fn`.
- Attention: rotary positional embeddings, often grouped-query attention with
  fewer key/value heads than query heads.
- Multiplicative gates: present.

Implementation checklist:

- Apply `LN-rule` to every Qwen RMSNorm using the epsilon attribute supplied
  by the model implementation.
- Apply `Identity-rule` to `act_fn(gate_proj(x))`.
- Apply `Half-rule` to the product
  `act_fn(gate_proj(x)) * up_proj(x)` before `down_proj`.
- When applying `AH-rule` in HuggingFace, load with
  `attn_implementation="eager"`. Flash attention or SDPA paths often bypass
  the Python function that RelP needs to patch.
- In grouped-query attention, repeat key/value heads the same way the model's
  attention code does, usually via `repeat_kv` and `num_key_value_groups`,
  before computing the softmax pattern to detach.
- Because Qwen is rotary, do attention attribution after RoPE has been applied
  to queries and keys. In TransformerLens this means using rotated query/key
  hooks or post-softmax pattern/output hooks, not raw pre-RoPE `hook_q` and
  `hook_k`.

HuggingFace-style setup:

```python
from transformers import AutoModelForCausalLM
from relp import relp_mode

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    attn_implementation="eager",
    torch_dtype="auto",
    device_map="auto",
)

with relp_mode(model, rules=["LN", "Identity", "Half", "AH"]):
    metric.backward()
```

### Gemma / Gemma 2 family

Architecture facts:

- Normalization: Gemma-style RMSNorm.
- MLP: gated activation MLP.
- Attention: rotary positional embeddings. Gemma 2 also uses attention-logit
  soft-capping.
- Multiplicative gates: present.

Implementation checklist:

- Apply `LN-rule` to Gemma RMSNorm, but preserve Gemma's additive weight form:
  the normalized hidden state is multiplied by `(1 + weight)`, not just
  `weight`.
- Apply `Identity-rule` to the gated MLP activation.
- Apply `Half-rule` to the gated product before the down projection.
- In Gemma 2 attention, preserve the tanh soft-cap path before mask and
  softmax. Detach the softmax probabilities for `AH-rule`; do not detach or
  remove the soft-cap transformation itself.
- Because Gemma is rotary, compute attention relevance after RoPE. Prefer
  residual stream, MLP output, attention output, or per-head `z` scores to raw
  pre-RoPE query/key attribution.

### Llama / Mistral-style models

Architecture facts:

- Normalization: RMSNorm.
- MLP: gated `gate_proj`, `up_proj`, `down_proj`.
- Attention: rotary; grouped-query attention is common.
- Multiplicative gates: present.

Implementation checklist:

- Use the Qwen-style gated MLP rules: `LN`, `Identity`, and `Half`.
- Use eager attention for `AH-rule`.
- Handle grouped-query attention by repeating key/value heads before softmax.
- Avoid comparing pre-RoPE query/key vectors across positions.

## Implementation paths

There is no native `autointerp.tools.relp` wrapper yet. Use this skill to
implement or call an external RelP runner, and use the existing
`gradient-attribution`, `attribution-patching`, and `activation-patching`
tools for baselines and confirmation.

### Modified TransformerLens path

Use the official RelP TransformerLens fork when you want hook names and cache
semantics close to existing circuit-analysis workflows:

```python
model.cfg.use_lrp = True
model.cfg.LRP_rules = ["LN-rule", "Identity-rule", "Half-rule"]
```

Relevant classes/functions in that fork:

- `LayerNorm`, `LayerNormPre`, `RMSNorm`, `RMSNormPre`: detach scale under
  `LN-rule`.
- `ModifiedAct` and `CanBeUsedAsMLP.select_activation_function`: implement
  `Identity-rule`.
- `GatedMLP.forward` and `GatedMLP4Bit.forward`: implement `Half-rule`.
- `AbstractAttention.forward` and `GroupedQueryAttention.calculate_z_scores`:
  detach attention pattern under `AH-rule`.

### HuggingFace monkey-patch path

Use the Pando-style standalone patcher when the target is a normal
`AutoModelForCausalLM`:

```python
state = relp_patch(model, rules=["LN", "Identity", "Half", "AH"])
try:
    metric.backward()
finally:
    relp_unpatch(model, state)
```

Implementation details to preserve:

- Classify modules before patching and fail on unknown modules. Silent fallback
  to normal gradients defeats the point of RelP.
- Patch normalization module `forward` methods, not only backward hooks.
- Patch gated and ungated MLPs differently.
- Patch module-level `eager_attention_forward` functions for `AH-rule`; this
  is why eager attention is required.
- Restore all patched functions after each run, especially in long-lived
  agents or notebooks.
- Merge or account for LoRA adapters before patching if wrapper modules hide
  the underlying `Linear`, MLP, or attention classes.

## Scoring and reporting

Report:

- model id, architecture class, tokenizer/chat template, and precision;
- original and patch prompts, including token alignment checks;
- scalar metric and target token ids;
- component type and hook names;
- exact RelP rules enabled;
- whether attention used eager, SDPA, or flash attention;
- top signed and absolute sites;
- comparison with raw gradient attribution and, for top sites, activation
  patching or ablation.

For Pando-style field attribution, aggregate token or component scores back to
input fields only after preserving raw token scores. Report field-level scores
with labels, but keep enough token-level evidence to debug tokenization and
template artifacts.

## Evidence standards

Strong RelP evidence has all of:

- matched original/patch inputs and an explicit metric;
- model-appropriate rules for normalization, activations, gates, and attention;
- stable rankings across prompt paraphrases or held-out examples;
- agreement with activation patching or ablation on top-ranked sites;
- a baseline showing that RelP improves on or clarifies standard gradient
  attribution.

Moderate evidence is a stable RelP ranking with correct rules and a plausible
baseline, but no activation-patching confirmation.

Weak evidence is a single heatmap, missing rule metadata, use of the wrong
attention implementation, or scores from a model family whose modules were not
classified.

## Common failure modes

- Applying `Half-rule` to ungated GPT-2 or Pythia MLPs.
- Forgetting `Half-rule` for Qwen, Gemma, Llama, or Mistral gated MLPs.
- Treating RMSNorm as LayerNorm and accidentally subtracting a mean.
- Forgetting Gemma's `(1 + weight)` RMSNorm convention.
- Running `AH-rule` while the model uses flash attention or SDPA.
- Detaching pre-softmax attention scores instead of post-softmax
  probabilities.
- Ignoring grouped-query attention and using unrepeated key/value heads.
- Comparing raw pre-RoPE `q`/`k` across positions in rotary models.
- Letting monkey-patches persist after the RelP run.
- Treating RelP as causal proof instead of a triage method.

## Related skills

- `gradient-attribution` for the raw-gradient baseline.
- `attribution-patching` for the first-order patching framing.
- `activation-patching` and `causal-validation` for confirmation.
- `attention-heads` when RelP localizes to attention output or per-head `z`.
- `sparse-autoencoders` when applying RelP to SAE feature circuits.

## References

- RelP paper: https://arxiv.org/abs/2508.21258
- Official RelP code: https://github.com/FarnoushRJ/RelP
- Pando benchmark and RelP agent: https://ar-forum.github.io/Pando/
- Pando code: https://github.com/AR-FORUM/Pando
- AuditBench context for hidden-behavior auditing: https://arxiv.org/abs/2602.22755
