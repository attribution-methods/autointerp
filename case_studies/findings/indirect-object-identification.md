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

## After the optimization pass
**Verdict: SUPPORTED.** The diagnosis was a chain of two scaffold bugs, both fixed generally (no answer hardcoding): the agent posed IOI as a `"who is the indirect object?"` quiz to a base model, and `format_messages` then crashed on gpt2 (it gated on `hasattr(tok, "apply_chat_template")`, true even though no template is set), so *every* generation was the string `<error: chat_template not set>` and accuracy was 0.0. With the chat-template fix plus planner guidance to use the standard IOI **completion** paradigm and an adequate sample size, the re-run framed IOI as next-token completion ("Mary gave a book to →", target-vs-foil logit) over **200** real `model_forward` trials and reached `hit_rate = 1.0` (≥ 0.6 gate), completing as SUPPORTED with zero chat-template errors and zero placeholder stubs.

**Honest gap:** this confirms IOI *behaviorally* but does not localize the name-mover circuit — the agent gated on `hit_rate` and never made the localization stage the gate. The multi-head patching now needed for that is push-button (`circuit_recovery_capture`; on gpt2 IOI the single best name-mover head recovers 0.12 vs ~0.97 for the top-3 set), but driving the weak planner to actually localize is the next round.
