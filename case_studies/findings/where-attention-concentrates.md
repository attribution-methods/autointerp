# Where Generation-Time Attention Concentrates

**Question:** When a model generates text, where does attention tend to concentrate, and why?
**Run:** model `gpt2-medium`, driver gpt-5-nano
**Verdict:** INCONCLUSIVE

## What the agent did
The agent drafted a four-stage plan (setup -> black_box -> localization -> validation) on GPT-2 Medium over 24 generated prompts, hypothesizing that the causal mask drives attention mass onto recent tokens. It registered one criterion, `recency_bias_strength`: the metric `logit_diff` (top-1 minus top-2 next-token logit) should exceed 0.5 on at least 4 dev prompts, and a patch suppressing attention to the last k tokens should then lower it. After approval it cached activations, ran forward passes to capture per-prompt logits as provenance, and computed `logit_diff` across stages.

## What it found
The agent measured `logit_diff` of 2.088 (above the 0.5 threshold) and read this as a positive recency signal, but the planned patch was never applied in the validation stage. The criterion was therefore recorded INCONCLUSIVE with reason `patch_not_applied_in_stage3`: with no patched comparison, the agent could not show the predicted drop, so the "why" (causal role of last-token attention) went untested. The run reported 0/1 criteria passed.

## Takeaway (testbed reach)
The testbed cleanly produced a provenance-backed correlational metric, but the causal validation stage silently skipped its perturbation, so the run measured a signal without confirming the mechanism behind it.
