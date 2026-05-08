# Investigation Log

Agent narrative notes for this run.

## Stage 0: Black Box Evaluation (COMPLETED)

**Date:** 2026-05-07  
**Objective:** Confirm GPT-2-small has a strong IOI signal on the dev split before proceeding to expensive discovery loop.

### Results

Ran forward passes on 500 IOI dev pairs generated with seed=42.

**Metrics:**
- **Accuracy:** 0.98 (490/500 samples where logit(IO) > logit(S))
- **Logit Diff:** 2.6725 (mean difference between IO and S logits)
- **Distribution:** 
  - Min logit_diff: -1.0
  - Max logit_diff: 6.5
  - Std logit_diff: 1.21

**Criterion Evaluation:**
- ✅ **behavioral-sanity** PASSED: accuracy 0.98 >= 0.95 threshold

### Analysis

GPT-2-small demonstrates a very strong IOI signal:
1. On 98% of dev examples, the model assigns higher logit to the indirect object (IO) than to the subject (S)
2. The mean logit difference of 2.67 is well above the 2.0 abort threshold
3. Only 10/500 samples fail to prefer IO over S

This confirms the model has learned the IOI task and justifies proceeding to the feature discovery stage to identify the mechanistic circuit.

### Artifacts
- `scratch/stage0_blackbox_results.json` - Full results
- `scratch/ioi_dev_pairs.json` - Generated IOI pairs for reuse
- `findings/stage_0_black_box/metric_stage0-dev-accuracy.json` - Accuracy metric result
- `findings/stage_0_black_box/metric_stage0-dev-logit-diff.json` - Logit diff metric result

---

## Stage 1: Feature Discovery (COMPLETED)

**Date:** 2026-05-07  
**Objective:** Use the discover_features sub-agent to identify candidate attention heads for IOI, then validate with joint patching.

### Discovery Sub-Agent (Step A)

Called `discover_features()` with max_iterations=4. The sub-agent iterated through 4 algorithm candidates, evaluating attention heads on a small IOI benchmark (n_pairs=30).

**Best Algorithm:** `algorithm_v4`
- **Per-algorithm recovery:** 0.182 (single-head baseline on small benchmark)
- **Top 10 heads by score:**
  1. L9H9 (score: 29.0)
  2. L10H7 (score: 23.0)
  3. L9H6 (score: 18.0)
  4. L11H10 (score: 17.25)
  5. L10H0 (score: 14.44)
  6. L10H10 (score: 12.06)
  7. L10H6 (score: 11.19)
  8. L10H2 (score: 9.94)
  9. L11H3 (score: 9.25)
  10. L10H1 (score: 8.69)

**Cost:** 4 iterations, $1.19 USD, ~15 minutes

### Joint Top-K Patch on Dev (Step B)

After discovery, extracted top-K candidate heads and performed **simultaneous** joint patching on the full dev split (500 pairs).

**Baseline:**
- Clean logit_diff: 2.67
- Corrupt logit_diff: -3.80
- Gap: 6.47

**K Sweep Results:**
- K=3: recovery = 0.388 (below threshold)
- K=4: recovery = 0.256 (worse - head L11H10 hurts when added to top-3)
- **K=5: recovery = 0.456** ✅ (above 0.4 threshold)
- K=6: recovery = 0.532
- K=7: recovery = 0.601
- K=8: recovery = 0.675

**Selected K=5** to meet the discovery-recovery criterion threshold of 0.4.

**Final Top-5 Circuit:**
1. L9H9 (layer 9, head 9)
2. L10H7 (layer 10, head 7)
3. L9H6 (layer 9, head 6)
4. L11H10 (layer 11, head 10)
5. L10H0 (layer 10, head 0)

**Joint Recovery:** 0.456 (recovers 45.6% of clean-corrupt gap)

### Criterion Evaluation

- ✅ **discovery-recovery** PASSED: patch_effect_recovery 0.456 >= 0.4 threshold

### Analysis

The discovery sub-agent successfully identified a compact set of attention heads that implement IOI:
- The top-5 heads span layers 9-11 (late in the 12-layer model)
- Joint patching these 5 heads recovers nearly half of the clean-corrupt gap
- Head combinations exhibit non-monotonic behavior (K=4 worse than K=3), suggesting complex interactions
- The discovered circuit is concentrated in a narrow layer range, consistent with the hypothesis that IOI involves late-layer name-mover heads

### Artifacts
- `discovery/stage01_call6/` - Discovery sub-agent session directory
- `scratch/stage1_joint_patch_final.json` - K=5 joint patch results
- `scratch/stage1_K_sweep.json` - Full K sweep results
- `findings/stage_1_feature_discovery/metric_stage1-dev-joint-patch-recovery.json` - Recovery metric
- `findings/stage_1_feature_discovery/candidate_*.json` - 5 CandidateSite artifacts

---

## Stage 2: Localization (COMPLETED)

**Date:** 2026-05-07  
**Objective:** Validate discovery results with exhaustive head_patch_sweep across all 144 (layer, head) pairs, then run joint top-K patch.

### Exhaustive Head Patch Sweep

Ran head_patch_sweep on all 12 layers × 12 heads = 144 sites, patching each head individually from clean to corrupt runs.

**Top 10 Heads by Individual Recovery:**
1. L9H9: 0.499 ⭐ (discovery rank #1)
2. L9H6: 0.222 ⭐ (discovery rank #3)
3. L10H0: 0.181 ⭐ (discovery rank #5)
4. L10H10: 0.080
5. L10H2: 0.077
6. L11H2: 0.070
7. L10H6: 0.063
8. L9H8: 0.056
9. L11H1: 0.046
10. L11H3: 0.046

**Key Finding - Discovery vs Localization Discrepancy:**
- Discovery rank #2 (L10H7): recovery = **-0.239** (ranks 144/144 - WORST!)
- Discovery rank #4 (L11H10): recovery = **-0.114** (ranks 143/144)

These 2 heads from discovery actually **hurt** performance when patched individually! This reveals a critical limitation of the discovery sub-agent's small-benchmark scoring.

**Overlap Analysis:**
- 3/5 discovery heads appear in localization top-10: (L9H9, L9H6, L10H0)
- Discovery-only: (L10H7, L11H10) - both have negative individual recovery!
- Localization-only top-10: (L10H10, L10H2, L11H2, L10H6, L9H8, L11H1, L11H3)

### Joint Top-K Patch with Localization Heads

Using the localization top-K heads, tested joint patches:

**K Sweep Results:**
- K=1 (just L9H9): recovery = 0.515
- K=2 (L9H9, L9H6): recovery = 0.775
- **K=3 (L9H9, L9H6, L10H0): recovery = 0.990** ✅ (99%!)
- K=4+: over-recovery (>1.0)

**Final Localization Circuit (K=3):**
1. L9H9 (layer 9, head 9)
2. L9H6 (layer 9, head 6)
3. L10H0 (layer 10, head 0)

**Joint Recovery:** 0.990 (recovers 99% of clean-corrupt gap!)  
**KL(patched || clean):** 0.415 nats (very close to clean distribution)

### Criterion Evaluation

- ✅ **localization-recovery** PASSED: patch_effect_recovery 0.990 >= 0.85 threshold

### Analysis

The exhaustive localization sweep reveals a **much more compact and effective circuit** than discovery:

**Comparison:**
- Discovery: 5 heads, 45.6% recovery
- Localization: 3 heads, 99.0% recovery

**Key Insights:**
1. **L9H9 is the dominant head** (0.499 individual recovery) - nearly half the gap alone
2. **L9H6 and L10H0 are supporting heads** - together with L9H9, they achieve near-perfect recovery
3. **Discovery false positives:** L10H7 and L11H10 were ranked #2 and #4 by discovery but have strongly negative individual effects. This suggests:
   - The small discovery benchmark (n=30) doesn't generalize well
   - These heads may have indirect effects that only appear in specific contexts
   - Discovery's scoring was misled by noise or overfitting

4. **All 3 localization heads are in layers 9-10** (late layers, consistent with IOI hypothesis about name-mover heads at the final prediction stage)

**Conclusion:**  
The localization stage successfully identified a minimal, high-fidelity IOI circuit. The 3-head circuit (L9H9, L9H6, L10H0) explains virtually all of GPT-2-small's IOI behavior on the dev split.

### Artifacts
- `scratch/stage2_head_sweep_results.json` - Full 12×12 recovery matrix
- `scratch/stage2_joint_patch_localization.json` - K sweep results
- `scratch/stage2_metrics.json` - Final metrics with K=3
- `findings/stage_2_localization/metric_stage2-dev-localization-recovery.json` - Recovery metric
- `findings/stage_2_localization/metric_stage2-dev-kl-to-clean.json` - KL metric

---
