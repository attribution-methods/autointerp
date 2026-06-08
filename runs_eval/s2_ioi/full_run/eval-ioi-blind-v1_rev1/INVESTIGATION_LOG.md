# Investigation Log: eval-ioi-blind-v1_rev1

## Stage 0: Black-Box Behavioral Measurement (COMPLETED)

**Goal:** Confirm GPT-2-small has a strong IOI behavioral signal.

**Dataset Generated:**
- Created IOI dataset with 500 dev samples and 100 heldout samples
- Balanced 50/50 ABBA/BABA templates
- Single-token names, places, and objects
- ABC corruption scheme (replace duplicate name with novel name C)

**Results:**
- **Accuracy: 99.80%** (499/500 samples correct)
  - Criterion: behavioral-sanity ✅ PASSED (0.998 >= 0.95)
- **Mean logit_diff: 2.92** (logit(IO) - logit(S))
  - Well above the 2.0 abort threshold

**Key Findings:**
- GPT-2-small strongly prefers the indirect object (IO) name over the subject (S) name
- Only 1 out of 500 samples failed (99.8% accuracy)
- Mean logit difference of 2.92 provides strong signal for patching experiments

**Abort Checks:**
- ✅ Accuracy >= 0.9: PASS
- ✅ Logit_diff >= 2.0: PASS

**Next Steps:**
- Proceed to Stage 1: Localization
- Run logit lens analysis to see where IO information emerges
- Perform activation patching across all (layer, head) pairs

---

## Stage 1: Localization (COMPLETED - CRITERION FAILED)

**Goal:** Identify which attention heads causally implement IOI via activation patching.

### Methods

1. **Logit Lens Analysis** (50 dev samples)
   - Measured KL divergence from final layer across all 12 layers at END position
   - Tracked when IO token appears in top-10 predictions

2. **Comprehensive Head Patching Sweep** (all 500 dev samples)
   - Patched clean activations INTO corrupt (ABC) runs for all 144 heads (12 layers × 12 heads)
   - Measured recovery of logit_diff at END position
   - Used head-level granularity (patching at c_proj output)

### Results

**Logit Lens:**
- IO information emerges gradually in layers 7-11
- Layer 9: mean KL = 75.0, IO token occasionally in top-10
- Final layers (10-11) show massive KL divergence increase (144, 560)

**Head Patching Recovery (Top 10 Heads):**
| Rank | Head  | Recovery |
|------|-------|----------|
| 1    | L9H9  | 0.4634   |
| 2    | L8H6  | 0.4413   |
| 3    | L8H10 | 0.2436   |
| 4    | L7H9  | 0.2209   |
| 5    | L10H0 | 0.1417   |
| 6    | L9H7  | 0.1036   |
| 7    | L7H3  | 0.0895   |
| 8    | L10H6 | 0.0763   |
| 9    | L10H10| 0.0592   |
| 10   | L10H1 | 0.0438   |

**Statistics:**
- Mean recovery across all 144 heads: 0.0018
- Best single head (L9H9): 0.4634 (46.3%)
- Top-10 mean recovery: 0.1883 (18.8%)
- Clean logit_diff: 2.92, Corrupt: 1.53, Gap: 1.40

### Key Findings

1. **Distributed Implementation**: IOI is implemented by multiple heads in layers 7-10, not a simple 1-2 head circuit
2. **Layer 9 is Critical**: L9H9 and L9H7 show the strongest individual effects
3. **Layer 8 Co-Contributors**: L8H6 and L8H10 also show strong recovery (~44% and ~24%)
4. **No Single "Copy Head"**: Unlike some interpretability narratives, no single head achieves >50% recovery

### Criterion Failure

**localization-recovery: FAILED** (0.46 < 0.85 threshold)

The best single head (L9H9) achieves 46% recovery, far below the 85% threshold. Joint patching of multiple top heads showed non-additive interactions (negative in some configurations), suggesting complex dependencies between components.

### Spec Revision Request

Requested spec revision because:
1. The 85% threshold appears too high for GPT-2-small's actual IOI implementation
2. The criterion may conflate "recovery" (additive patching) with "faithfulness" (complement ablation)
3. Prior IOI work may have used different models, datasets, or patching protocols

**Recommendation:** Adjust threshold to ~0.45-0.50 (matching observed best-head performance), or clarify that faithfulness testing via complement ablation is the intended measurement.

---

## Investigation Terminated

Status: REVISION_REQUESTED
Reason: localization-recovery criterion failed (0.46 < 0.85)
Next: Spec author should review findings and revise success criteria or hypothesis
