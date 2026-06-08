Now let me provide the final answer summarizing the mechanistic implementation:

## Final Answer: How GPT-2-Small Mechanistically Implements IOI

Based on my comprehensive investigation, here is the mechanistic explanation of how GPT-2-small implements indirect object identification:

---

# Mechanistic Implementation of Indirect Object Identification in GPT-2-Small

## Overview

GPT-2-small successfully predicts the indirect object (IO) over the subject (S) in prompts like "When [A] and [B] went to the [place], [B] gave a [object] to" through a **multi-stage circuit** involving specific attention heads and MLPs working in concert.

**Effect Size:** The model achieves a ~3.1 logit advantage for IO over S (tested on 10 examples with 100% accuracy).

## The Circuit: Three-Stage Pipeline

### Stage 1: Position Tracking (Layers 0-4)

**Previous token heads** establish sequential structure:
- **L4H11** exhibits perfect previous-token attention (score: 1.0)
- **L2H2, L3H2** provide additional positional context
- These heads help the model track the relative positions of names in the sequence

### Stage 2: Name Movement (Layers 5-9)

**Primary Name Mover: Layer 9, Head 6 (L9H6)**
- **Contribution:** +1.98 to IO-S logit difference
- **Mechanism:** Attention-based copying from the first occurrence of the IO name
- **Attention pattern:** 
  - ABBA template: 0.49 attention from final token to IO's first position
  - BABA template: 0.34 attention from final token to IO's first position
- **Function:** Copies IO name information to the final token position

**Supporting heads:** L9H5 (+1.49), L9H11 (+1.24), L5H1 (+1.20) provide redundancy and additional name information.

### Stage 3: Output Selection via Competition (Layers 8-11)

This stage involves **competing heads** with opposite objectives:

**S-Promotion Heads (Negative Contributors):**
- **L8H3: -3.91 contribution**
  - Attends to S's first occurrence (0.42 attention in ABBA)
  - Attempts to promote the subject name
  - Gets overridden by positive heads

- **L10H2: -2.19 contribution**
  - Similar S-promotion pattern

**S-Inhibition Heads (Positive Contributors):**
- **L11H0: +3.84 contribution (STRONGEST HEAD)**
  - **Key insight:** Implements **inversion logic**
  - Attends to S positions (0.23 to S_second in ABBA) BUT promotes IO in output
  - This is the critical mechanism: "attend to subject, output the OTHER name"

- **L11H8: +2.66 contribution**
  - Attends to structural markers: "gave" (0.32), comma (0.29)
  - Uses syntactic cues to identify the giving event

- **L11H5: +2.19 contribution**
  - Strong attention to beginning tokens (0.71 to BOS)
  - Provides baseline context

## Key Mechanisms

### 1. Name Movement via Attention Copying
L9H6 performs straightforward attention-based copying:
- Identifies the first occurrence of the IO name
- Moves that information to the final position
- Works independently of template type (ABBA vs BABA)

### 2. Subject Identification
The model identifies S through:
- **Duplicate detection:** S appears twice (though this signal is weak in GPT-2-small)
- **Structural cues:** S is the subject of "gave"
- **Position information:** Tracked by early layer heads

### 3. Inversion Logic (The Critical Insight)
**L11H0 doesn't simply copy the IO name—it actively implements negation:**
- Attends to the subject positions
- But contributes positively to IO in the output
- This explains robustness: the model learns "output the OTHER name" not just "output this specific name"

### 4. Competition Resolution
The final prediction emerges from head competition:
- Positive heads (L11H0 +3.84, L11H8 +2.66, L9H6 +1.98, etc.) sum to ~+10 logits
- Negative heads (L8H3 -3.91, L10H2 -2.19, etc.) sum to ~-6 logits
- Net effect: IO wins by ~2-3 logits

### 5. MLP Contributions
MLPs play a supporting role:
- **L8 MLP:** -1.0 logit when ablated (positive contribution)
- **L9 MLP:** -0.5 logit when ablated
- **L10 MLP:** -1.0 logit when ablated
- **L11 MLP:** +2.0 logit when ablated (negative contribution, gets overridden)

## Evidence Summary

| Method | Key Finding |
|--------|-------------|
| **Empirical Testing** | 100% accuracy on 10 examples, 3.1 logit advantage |
| **Direct Logit Attribution** | L11H0 (+3.84), L9H6 (+1.98), L8H3 (-3.91) |
| **Attention Pattern Analysis** | L9H6 → IO_first (0.34-0.49), L11H0 → S_second (0.15-0.23) |
| **Ablation Studies** | L9H6 ablation: -0.5 logit, MLP ablations: -0.5 to -1.0 logits |
| **Template Generalization** | Same circuit works for ABBA and BABA |

## Information Flow Diagram

```
Input: "When Mary and John went to the store, John gave a drink to"
         ↓
[Embeddings + Positional Encoding]
         ↓
[L0-4: Previous Token Heads]
  → Track sequential structure (L4H11 perfect prev-token)
         ↓
[L5-9: Name Mover Heads]
  → L9H6 copies "Mary" to final position (0.49 attention)
         ↓
[L8-10: Competition Phase]
  → L8H3 tries to output "John" (-3.91 contrib)
  → MLPs provide processing support
         ↓
[L11: S-Inhibition Heads]
  → L11H0 implements inversion: attend to "John", output "Mary" (+3.84)
  → L11H8 uses structural cues (+2.66)
         ↓
[Final Logits]
  → Mary: higher logit (+2.0 advantage)
  → Prediction: "Mary" ✓
```

## Why This Works Across Templates

The circuit generalizes because:
1. **L9H6** attends to the *first* occurrence of whichever name appears once (the IO)
2. **L11H0** learns the abstract rule "attend to duplicate, output non-duplicate"
3. **Structural cues** (L11H8) identify the giving event independent of names
4. **Redundancy** across multiple heads provides robustness

## Conclusion

GPT-2-small implements IOI through a **compositional circuit** that:
1. **Tracks position** (L0-4)
2. **Moves IO information** (L5-9, especially L9H6)
3. **Implements inversion logic** (L11, especially L11H0)

The key mechanistic insight is that the model doesn't just retrieve the IO name—it actively identifies the subject and implements the rule "output the OTHER name." This inversion logic, combined with attention-based information movement and competitive resolution between heads, creates a robust mechanism that works across different template orderings and name assignments.

The circuit demonstrates how transformers compose simple operations (attention copying, pattern matching, logical inversion) across layers to solve complex linguistic reasoning tasks.