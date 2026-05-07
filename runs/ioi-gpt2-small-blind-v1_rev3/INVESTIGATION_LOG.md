# Investigation Log

Agent narrative notes for this run.

## Stage 0: Black-box Behavioral Signal (COMPLETED)

Measured accuracy and logit_diff on 500 IOI dev samples:
- **Accuracy: 0.98 (98%)** - exceeds 0.95 threshold
- **Mean logit_diff: 2.67** - strong contrastive signal

Behavioral-sanity criterion: **PASSED**

## Stage 1: Localization via Head Patching (COMPLETED)

Ran two-pass localization:
1. Per-head sweep across all 144 (layer, head) pairs
2. Identified top-3 heads: **(9,9), (9,6), (10,0)**
3. Joint recovery by patching all 3 heads simultaneously: **0.93**

Metrics:
- patch_effect_recovery (joint): **0.93**
- kl_to_clean: **0.36**

Localization-recovery criterion: **PASSED** (0.93 >= 0.85)

The circuit consists of:
- Layer 9, Head 9 (primary, 0.50 individual recovery)
- Layer 9, Head 6 (secondary, 0.22 individual recovery)  
- Layer 10, Head 0 (tertiary, 0.18 individual recovery)


## Stage 3: Intervention (COMPLETED)

Path patching and faithfulness measurement on dev split:
- Circuit C = [(9,9), (9,6), (10,0)]
- Faithfulness (dev): **0.51**
- Patch effect recovery: **0.93**

## Stage 4: Validation on Heldout (FAILED CRITERIA)

Evaluated circuit on 100 heldout samples (disjoint names):
- Clean logit_diff: **3.22**
- Corrupt logit_diff: **-4.72**
- Circuit-only logit_diff: **-0.63**
- Faithfulness: **0.52** (threshold: 0.85) ❌
- Minimality: **0.00** (unclipped: -0.04, threshold: 0.05) ❌

### Key Findings

**Success:**
- Circuit achieves 0.93 joint recovery (clean → corrupt patching)
- Behavioral-sanity and localization-recovery criteria passed
- Clean baseline performance strong (98% accuracy, 2.67 logit_diff)

**Failure:**
- Circuit-only performance (0.52 faithfulness) far below 0.85 threshold
- Negative minimality suggests antagonistic head interactions
- Gap between recovery (0.93) and faithfulness (0.52) indicates circuit incompleteness

### Spec Revision Requested

The 3-head circuit is causally important for IOI but insufficient for faithful reproduction. Possible causes:
1. Circuit size too small (Wang et al. may have used 5-7 heads)
2. END-only mean-ablation of 141 heads still too harsh
3. Missing supporting heads below top-3 in sweep
4. Implementation differences from Wang et al. 2022

Recommendation: Expand circuit (try top-5/7), validate implementation, or adjust threshold to empirical ~0.5-0.6 range for 3-head circuits.

