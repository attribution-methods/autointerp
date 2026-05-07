# Scratchpad

## Task
Discover attention heads in GPT-2-small that causally implement indirect object identification (IOI). The task is to identify which (layer, head) pairs enable the model to prefer the indirect-object name (IO) over the subject name (S) at the prediction position 'to ___' in IOI prompts like 'When [A] and [B] went to the [place], [B] gave a [object] to'. Clean prompts have ABBA or BABA structure where IO appears once and S appears twice. Corrupt prompts swap IO with a novel name C (ABC corruption). Candidate heads should show high patch_effect_recovery when patching clean head activations into corrupt runs.

## Reward
Optimization target: `patch_effect_recovery` (higher is better).

## Status
[NEXT] Iteration 4: Implement path-patching attribution. algorithm_v3 (PER=0.248) achieved strong results by directly measuring patch_effect_recovery via head-patch-sweep. However, head patching measures the effect of changing one head while all other heads adapt in later layers. Path patching isolates the *direct* effect of each sender head on the final logits by freezing all other heads to their corrupt values. This should provide even more precise causal attribution and may identify heads whose direct contribution matters most for the IOI task.
