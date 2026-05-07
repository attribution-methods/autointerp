# Investigation Log

Agent narrative notes for this run.

## Stage 2: Localization Complete

### Discovery-Localization Agreement: TRUE ✓

**Discovery top-3 heads (from discover_features sub-agent):**
- L9H9 (score: 0.514)
- L9H6 (score: 0.244)  
- L10H0 (score: 0.192)

**Localization top-3 heads (from exhaustive head_patch_sweep):**
- L9H9 (recovery: 0.499)
- L9H6 (recovery: 0.222)
- L10H0 (recovery: 0.181)

**Result:** Perfect agreement! The discovery sub-agent successfully identified the canonical IOI circuit (Wang et al. 2022).

**Joint top-3 patch performance:**
- Patch effect recovery: 0.9903 (99.03%)
- KL(clean || patched): 0.416

The discovered circuit achieves near-perfect recovery of the clean-corrupt logit_diff gap, confirming these three attention heads form the core IOI mechanism in GPT-2-small.

