---
name: predictive-concept-decoders
description: Predictive Concept Decoder workflows for querying and auditing sparse concept bottleneck decoders over model activations. Use for Transluce-style PCD evidence, latent or unreported behavior detection, jailbreak or secret-hint awareness, user-attribute belief readouts, implanted concept tests, and comparisons against activation oracles, SAEs, probes, and causal validation.
---

# Predictive Concept Decoders

## What this skill is

Predictive Concept Decoders (PCDs) are trained interpretability assistants.
An encoder reads a subject model's activations and compresses each token to a
sparse list of active concepts. A decoder then reads only those concepts plus
a natural-language question and answers a question about the subject model's
behavior.

Use PCDs as behavior-predictive activation evidence. They are especially
useful when direct model self-report is unreliable, such as hidden shortcut
use, jailbreak awareness, latent user assumptions, or injected concepts.

## When to use

Use this skill when a trained PCD or compatible PCD-style decoder is available
for the subject model or activation stream, and the investigation needs a
question-conditioned readout of latent information in activations.

Good fits:

- The model may be using information it does not mention in its output.
- You need to ask "what latent concept, shortcut, or belief is active here?"
- You want a sparse, auditable bridge from activations to a natural-language
  answer.
- You are comparing decoder evidence against SAEs, activation oracles,
  direct prompting, probes, or causal interventions.

Poor fits:

- No compatible PCD checkpoint exists for the subject model, tokenizer,
  layer, and activation stream.
- You need a causal claim and have no intervention budget.
- The question can be answered more directly with black-box probes or simple
  activation patching.

## Inputs

Record these before querying the decoder:

- Subject model, tokenizer/chat template, and prompt text.
- Activation source: layer, component, token span, and whether activations are
  from the user message, assistant response, or a held-out prompt.
- PCD checkpoint metadata: read layer, write layer, trained top-k, concept
  dictionary size, and compatible model.
- Decoder question, answer format, sampling temperature, number of samples,
  and any top-k override.
- Active concept ids, scores, labels/descriptions, and max-activating examples
  if available.

## Workflow

1. **Pre-register the question.** Phrase one narrow behavior question before
   seeing decoder outputs. Prefer short-answer or multiple-choice formats.
2. **Capture compatible activations.** Use the exact subject model, tokenizer,
   layer, and token span expected by the PCD. Store an `ActivationCacheRef`
   or equivalent manifest.
3. **Run the PCD.** Query with the trained top-k first. If the result is weak,
   vary top-k as a sensitivity check, but mark out-of-training settings as
   lower-confidence evidence.
4. **Audit active concepts.** Inspect the concept ids and descriptions that
   support the decoder answer. Check whether the answer is traceable to a
   small, coherent concept set rather than unsupported decoder text.
5. **Compare baselines.** Ask the subject model directly, ask it from a
   transcript, and compare with activation-oracle, SAE, probe, or no-bottleneck
   decoder baselines when available.
6. **Validate behaviorally.** Test whether the decoder answer predicts held-out
   behavior, counterfactual prompts, known labels, or intervention outcomes.
7. **Report uncertainty.** Separate "PCD predicts X from activations" from
   "X causally drives the behavior."

## Query design

Use concrete questions:

- "Which hidden hint or shortcut is the assistant using?"
- "Is the assistant assuming a particular value of this user attribute?"
- "What concept is active in the steered activation?"
- "What behavior is the assistant about to produce?"
- "Is the refusal more associated with user safety, policy compliance, or
  self-protection?"

Run multiple phrasings and samples. Report the answer distribution instead of
cherry-picking one completion. Avoid leading questions unless the task is a
pre-registered multiple-choice classification.

## Evidence standards

Strong PCD evidence has all of:

- The answer is stable across prompt phrasings, samples, and nearby top-k
  settings.
- The answer traces to active concepts with relevant descriptions or
  max-activating examples.
- Held-out prompts or labels confirm the predicted behavior.
- A control condition removes the concept or behavior.
- A causal intervention, when feasible, changes the downstream metric in the
  predicted direction.

Moderate evidence: stable decoder answer plus coherent active concepts, but
no causal test.

Weak evidence: a single decoder completion, no concept audit, no controls, or
a result that appears only under an out-of-distribution top-k/no-bottleneck
setting.

## Common investigation patterns

### Hidden shortcut or hint use

Feed activations from the hinted user message or pre-answer context. Ask
whether the assistant is using a shortcut, hidden hint, or external clue. Use a
control prompt with the hint removed, and judge whether the PCD answer changes
while direct self-report stays uninformative.

### Jailbreak or hidden-behavior awareness

Feed activations before or during the subject model's response. Ask what
behavior or topic is active. Compare to direct self-report and third-person
transcript prompting. Do not repeat unsafe procedural content in reports;
summarize the harmful category instead.

### User-attribute belief readout

Use multiple-choice questions and held-out attributes. Treat the answer as a
claim about the subject model's belief or assumption, not as a fact about the
user. Avoid exposing sensitive attributes except in controlled evaluations.

### Implanted or steered concept tests

Patch or steer a known concept vector into the activation stream, then ask the
PCD whether a specific concept is active. Compare against subject-model
introspection and against SAE or concept-neighbor descriptions. This tests
whether the PCD can verbalize latent activation content, not whether that
content naturally occurs.

### Surprising refusal or error explanations

When the decoder gives an unexpected explanation, audit the top concepts before
accepting it. Agreement between decoder output and independently generated
concept labels is corroborating evidence; a causal test is still needed for a
mechanistic claim.

## Training a new PCD

Training is a major model-building project, not a normal investigation step.
Use the original paper's setup as a reference point, not a universal recipe:

- Split web text into prefix, middle, and suffix.
- Run the subject model on prefix plus middle and read middle-token
  activations.
- Train an encoder with a linear concept dictionary and top-k sparsity.
- Re-embed active concepts as soft tokens for a decoder initialized from the
  subject model plus LoRA.
- Train the decoder to predict suffix tokens from encoded middle activations
  and suffix-so-far context.
- Use an auxiliary loss or other dead-concept mitigation; inactive concepts
  tend to have lower interpretability.
- Freeze the encoder before task-specific decoder finetuning unless the
  experiment explicitly tests encoder updates.
- Evaluate both predictive accuracy and concept interpretability on held-out
  data before using the PCD as evidence.

The Transluce reference experiment used Llama-3.1-8B-Instruct, 32,768
concepts, top-k 16, read layer 15, write layer 0, and 16-token
prefix/middle/suffix segments. Do not transfer these hyperparameters blindly
to a different model family.

## How to integrate with this repo

There is no native `autointerp.tools.pcd` wrapper yet. Until one exists,
write a small run script that:

1. Loads the subject model and activation cache.
2. Calls the external PCD checkpoint or hosted decoder.
3. Saves a JSON artifact with question, activation metadata, decoder samples,
   answer distribution, active concept ids, scores, labels, and controls.
4. Commits the result as a `BehavioralFinding`, `FeatureFinding`, or
   `CandidateSite`, then validates with `causal-validation` when possible.

Use related skills:

- `activation-cache` for compatible activation capture.
- `activation-oracles` for generic verbalizer comparisons.
- `sparse-autoencoders` and `pretrained-saes` for independent concept audits.
- `linear-probes` for simple label readouts.
- `causal-validation` before final mechanistic claims.

## Cautions

- PCD answers are learned decoder outputs, not ground truth.
- The sparse bottleneck makes outputs auditable, but it can also omit
  behavior-relevant information.
- Changing top-k or removing the bottleneck can surface extra information but
  may be out of distribution for the decoder.
- Concept labels are auto-interp hypotheses. Inspect max-activating examples
  and local prompt activations before relying on them.
- Question wording matters; use multiple phrasings and report variability.
- A PCD trained for one model/layer/tokenization should not be treated as
  compatible with another without validation.
- Do not infer real user attributes from PCD output. Report only the subject
  model's represented assumption in controlled settings.

## References

- Transluce overview and hosted decoder: https://transluce.org/pcd
- Paper: https://arxiv.org/abs/2512.15712
- Demo: https://decoder.transluce.org
