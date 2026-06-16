# Tracking Entities and Their Attributes Across Context

**Question:** How does a model keep track of which entity has which item across a long context?
**Run:** model `gpt2-medium`, driver gpt-5-nano
**Verdict:** SUPPORTED

## What the agent did
The agent drafted a two-stage plan: Stage 0 black-box probing (blackbox_probe, direct_logit_attribution) gated on `logit_diff > 0.2`, then Stage 1 localization (activation_cache, logit_lens, activation_patching) to causally find where the mapping is stored. After approval it ran only Stage 0: it built ~60 synthetic entity-item prompts, ran gpt2-medium forward passes, captured inputs as model_forward provenance, and computed `logit_diff` by comparing the target entity's candidate-token logit against a foil entity's token logit on the dev split.

## What it found
Observed mean `logit_diff` was 0.6055 against the 0.2 threshold (n_min 40), so the single pre-registered criterion passed. The agent read this as the model ranking the correct entity above distractors by ~0.605 in logit space and judged the hypothesis SUPPORTED (1/1 criteria passed), while noting the prompts are synthetic and Stage 1 localization remains to be run.

## Takeaway (testbed reach)
This only measured a behavioral logit-margin on synthetic prompts; the localization/causal-patching stage that would actually probe a binding mechanism never executed, so "SUPPORTED" rests on a small black-box proxy rather than any identified internal component.
