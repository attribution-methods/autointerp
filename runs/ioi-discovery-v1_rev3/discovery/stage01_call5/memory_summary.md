# Research Memory

## Best Candidates
- `algorithm_v3`: patch_effect_recovery=0.2479
- `algorithm_v1`: patch_effect_recovery=0.1673

## Current Signal
- Ablation is currently stronger than steering. Consider algorithms that emphasize causal-suppression features.

## Recent Experiment Notes
- `algorithm_v1`; hypothesis: Baseline: rank attention heads by mean(|z_clean - z_corrupt|) at prediction position. Simple gradient-free approach that measures which heads differ most between clean and corrupt prompts.; conclusion: Reasonable baseline (0.167 PER) identifying late-layer heads (L9-L11). However, this metric is indirect - it measures activation difference, not causal effect on the target metric. Next: try direct head-patch-sweep to rank heads by their actual patch_effect_recovery.
- `algorithm_v3`; hypothesis: Direct head-patch-sweep ranking: rank heads by their actual patch_effect_recovery when patching clean z into corrupt runs. This aligns the scoring metric with the optimization objective, measuring causal effect rather than just activation differences.; conclusion: Strong improvement (0.248 vs 0.167, +48%) by directly optimizing the target metric. Top head L9H9 recovers 51% of the clean behavior. The direct measurement successfully identifies causally important heads. Next: try path patching to isolate direct effects on logits, which may be even more precise for ranking heads by their causal contribution.

## Next Move
- Start from the current best candidate and change one idea at a time.
