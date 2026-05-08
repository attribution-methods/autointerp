# Scratchpad

## Task
Identify attention heads in GPT-2-small that implement Indirect Object Identification (IOI). The model receives prompts like "When John and Mary went to the store, Mary gave a drink to" and must predict the indirect object (John) over the subject (Mary). Find the (layer, head) sites that causally contribute to this behavior by maximizing the recovery of logit_diff(IO, S) when patching from clean to corrupted runs.

## Reward
Optimization target: `patch_effect_recovery` (higher is better).

## Status
[DONE] Final iteration complete. algorithm_v4 (position-aggregated z-diff) achieved best performance: 0.1819 patch_effect_recovery, improving 8.7% over v1's baseline (0.1673). Multi-position aggregation captures IOI computation across duplicate detection and prediction better than single-position scoring.
