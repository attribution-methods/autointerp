# Research Memory

## Best Candidates
- `algorithm_v4`: patch_effect_recovery=0.1819
- `algorithm_v1`: patch_effect_recovery=0.1673
- `algorithm_v3`: patch_effect_recovery=0.0867

## Current Signal
- Ablation is currently stronger than steering. Consider algorithms that emphasize causal-suppression features.

## Recent Experiment Notes
- `algorithm_v1`; hypothesis: Baseline: rank heads by mean |z_clean - z_corrupt| at prediction position; conclusion: Baseline establishes 0.167 recovery. Focuses on late layers (9-11). Next: try direct causal measurement via head_patch_sweep.
- `algorithm_v3`; hypothesis: Layer-normalized z-diff: apply z-score normalization within each layer to make cross-layer scores comparable and prevent late-layer magnitude dominance; conclusion: Layer normalization hurt performance (0.087 vs v1's 0.167). This shows late layers genuinely matter more for IOI - normalizing away magnitude differences removed important signal. Top-3 now includes early layer L1H7, suggesting over-correction.
- `algorithm_v4`; hypothesis: Position-aggregated z-diff: aggregate |z_clean - z_corrupt| across last 4 positions instead of just the final token, capturing IOI computation across duplicate detection and prediction.; conclusion: Best performer (0.1819 vs v1's 0.1673)! Aggregating across multiple positions captures more IOI signal than single-position scoring. Still focuses on late layers 9-11 but with improved recovery.

## Next Move
- Start from the current best candidate and change one idea at a time.
