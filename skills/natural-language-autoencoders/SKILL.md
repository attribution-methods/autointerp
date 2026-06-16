---
name: natural-language-autoencoders
description: Natural Language Autoencoder workflows for translating residual-stream activations into text hypotheses with activation verbalizer and reconstructor checkpoints. Use for targeted activation reading, hidden-behavior audits, and hypothesis generation when a compatible released NLA pair exists.
---

# Natural Language Autoencoders

Natural Language Autoencoders (NLAs) are activation verbalizer and activation
reconstructor pairs. The activation verbalizer (AV) maps a residual-stream
activation vector to a natural-language explanation. The activation
reconstructor (AR) maps that explanation back to a vector so the explanation
can be scored by reconstruction quality.

## When To Use

Use NLAs when you need human-readable hypotheses about what a residual-stream
activation contains and you have a compatible NLA for the target model, layer,
and activation width. They are most useful after cheaper localization has
already narrowed the search to specific prompts, layers, and token positions.

Good triggers:

- a hidden behavior may be represented internally but not verbalized;
- activation patching, logit lens, probes, or SAEs found a small set of
  behavior-relevant token positions;
- you need natural-language hypotheses to design black-box probes or causal
  interventions;
- an activation-oracle style workflow would help, but you want AV/AR
  reconstruction scores rather than text alone.

Avoid NLAs as a first pass over every token. They are expensive because the AV
generates hundreds of tokens per activation and the released checkpoints are
large.

## Workflow

1. Establish the behavioral question with black-box probes and define the
   prompt set, target model, layer, and token selector before querying the NLA.
2. Cache residual-stream activations at the exact layer expected by the NLA
   checkpoint. Record prompt id, token string, token index, layer, component,
   vector norm, model id, and split.
3. Confirm compatibility with the checkpoint sidecar (`nla_meta.yaml`): model
   family, `d_model`, prompt template, injection token ids, and injection
   scale. Load these values from the sidecar instead of hardcoding them.
4. Run the AV on a small, preselected batch. Sample multiple descriptions for
   important activations if the AV is stochastic. Preserve raw outputs and any
   parse failures.
5. When an AR is available, reconstruct each explanation and score it against
   the original vector. Treat low reconstruction error or high cosine
   similarity as evidence that the text preserves direction information, not
   as proof that the text is semantically correct.
6. Cluster explanations by concrete content: topic, entity, goal, plan,
   evaluation-awareness cue, refusal motive, memorized pattern, or formatting
   feature. Downweight generic assistant-role, safety-policy, and prompt-shape
   explanations.
7. Convert the strongest clusters into testable hypotheses. Validate with
   held-out prompts, activation patching, ablations, steering, or independent
   methods before making a mechanistic claim.

## Released Checkpoints

At time of writing, public NLA checkpoints are released as AV/AR pairs for:

- Qwen2.5-7B-Instruct, layer 20, `d_model=3584`
- Gemma-3-12B-IT, layer 32, `d_model=3840`
- Gemma-3-27B-IT, layer 41, `d_model=5376`
- Llama-3.3-70B-Instruct, layer 53, `d_model=8192`

The Hugging Face collection is `kitft/nla-models`. Use the matching AV and AR
for the same base model and layer. Do not mix Qwen, Gemma, or Llama sidecars,
tokenizers, scales, or activation dimensions.

## Inputs

Provide the NLA workflow with:

- residual-stream activation vectors of shape `[d_model]`;
- exact target model id, layer index, component name, prompt id, token index,
  and token text;
- the NLA checkpoint ids for the AV and AR;
- the checkpoint sidecar metadata;
- prompt split tags so discovery and validation remain disjoint.

For released inference code, a parquet file with an `activation_vector` column
is the expected bulk input format.

## Tools

Use `../../src/autointerp/tools/activations.py` to capture and cache the
target vectors before calling an external NLA runner:

```python
from autointerp.tools.activations import cache_components

cache = cache_components(
    handle,
    prompts=prompts,
    specs=["L20RESID"],
    token_index=-1,
)
```

Use vector comparison utilities when you have AR reconstructions:

```python
from autointerp.tools.vectors import cosine_similarity

score = cosine_similarity(original_vector, reconstructed_vector)
```

There is no bundled NLA inference tool in this repository yet. Treat SGLang
serving, `nla_inference.py`, Neuronpedia, or another NLA client as an external
runner and record the command, checkpoint ids, sidecar hash, sampling
parameters, and outputs.

## Evidence

Strong NLA evidence has all of the following:

- explanations are specific enough to predict a follow-up test;
- repeated AV samples or neighboring token positions converge on the same
  hypothesis;
- AR reconstruction scores are reported and are not outliers for that run;
- the hypothesis survives held-out prompts or causal intervention;
- negative controls do not produce the same explanation cluster.

Weak evidence includes one vivid explanation, an unscored AV output, a
mismatched checkpoint/layer, a generic assistant-policy description, or an
output that indicates injection failure.

## Cautions

NLA text is model-generated evidence about an activation, not direct access to
ground truth. Natural language can project human concepts onto activations that
may be distributed, unfamiliar, or only partially captured by the AV/AR pair.

Common failure modes:

- wrong layer, wrong base model, wrong tokenizer, or wrong `d_model`;
- hardcoded injection token ids or injection scales instead of sidecar values;
- raw vectors injected without the checkpoint's expected L2 scale;
- Gemma embedding-scale mistakes;
- parse failures from truncated AV generations;
- treating low AR error as causal evidence;
- validating on the same prompts used to select the activation.

When the NLA output suggests safety-relevant internal states such as hidden
motivation, deception, or evaluation awareness, require independent
corroboration before reporting a claim.
