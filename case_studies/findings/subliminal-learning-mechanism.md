# Mechanism of Subliminal Learning

**Question:** How does subliminal learning work in LLMs?
**Run:** model `gpt2-medium`, driver gpt-5-nano
**Verdict:** paused for plan revision

## What the agent did
The agent reframed the question as whether implicit priors in GPT-2 Medium show up as a prompt-invariant behavior and can be causally localized. It pre-registered a two-stage plan: Stage 0 black-box probing with criterion `implicit_priors_consistency` (logit_diff > 0 across neutral prompts), then Stage 1 localization with `causal_link_via_patching` (patch_effect_recovery > 0). It ran a generated script to compute logit_diff and accuracy over a 200-sample dev split.

## What it found
Stage 0 passed: logit_diff on dev was 1.745 (> 0), so `implicit_priors_consistency` was marked PASS; accuracy was 0.0 and not used for pass/fail. Stage 1 never ran — the agent could not produce patch_effect_recovery because it had no end-to-end patching workflow yielding real `model_forward` captures (clean/corrupt/patched) that the metric protocol requires, so it paused and requested a spec revision to add patching automation or relax the metric.

## Takeaway (testbed reach)
The behavioral half (a stable logit_diff signal) was cheap to obtain on a 355M open-weights model, but the causal patching half stalled on tooling/provenance plumbing rather than on the model itself.
