# Indirect Object Identification Circuit

**Question:** How do models perform indirect object identification (IOI)?
**Run:** model `gpt2-medium`, driver gpt-5-nano
**Verdict:** NOT SUPPORTED

## What the agent did
The agent drafted a two-stage plan: Stage 1 black-box probing (direct "who is the indirect object?" questions) gated on `ioi_accuracy_dev >= 0.60`, then Stage 2 localization (logit_lens, activation_cache). After approval it ran only Stage 0: a forward pass over a tiny 4-prompt dev probe, captured the inputs/outputs as provenance, and computed top-1 accuracy against binary indirect-object labels.

## What it found
Observed accuracy was 0.25 against the 0.60 threshold — predictions were `[0,0,0,0]` versus labels `[1,0,1,1]`. The agent judged the hypothesis NOT supported (0/1 criteria passed) and terminated the run on the failed gate, noting the n=4 sample was too small and the label-extraction too brittle for robust conclusions.

## Takeaway (testbed reach)
This run only measured a behavioral accuracy gap on 4 prompts and localized no circuit components — Stage 2's logit_lens/activation_cache localization never executed because the Stage 1 gate failed first.
