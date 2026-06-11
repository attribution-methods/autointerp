"""
Create comprehensive circuit diagram and summary
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
import json
import numpy as np
from autointerp.tools.model import load_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("="*80)
print("IOI CIRCUIT ANALYSIS - COMPREHENSIVE SUMMARY")
print("="*80)

# Load previous results
with open('scratch/head_dla.json', 'r') as f:
    dla_results = json.load(f)

# Parse and categorize heads
positive_heads = []
negative_heads = []

for key, val in dla_results['head_contributions'].items():
    layer, head = key.replace('L', '').replace('H', ' ').split()
    layer, head = int(layer), int(head)
    if val > 1.0:
        positive_heads.append((layer, head, val))
    elif val < -1.0:
        negative_heads.append((layer, head, val))

positive_heads.sort(key=lambda x: x[2], reverse=True)
negative_heads.sort(key=lambda x: x[2])

print("\nTOP POSITIVE CONTRIBUTORS (Name Mover & S-Inhibition Heads):")
print(f"{'Layer':<8} {'Head':<8} {'Contribution':<15}")
print("-" * 50)
for layer, head, contrib in positive_heads[:10]:
    print(f"{layer:<8} {head:<8} {contrib:<15.3f}")

print("\nTOP NEGATIVE CONTRIBUTORS (S-Promotion Heads):")
print(f"{'Layer':<8} {'Head':<8} {'Contribution':<15}")
print("-" * 50)
for layer, head, contrib in negative_heads[:10]:
    print(f"{layer:<8} {head:<8} {contrib:<15.3f}")

# Create circuit diagram
print("\n" + "="*80)
print("CIRCUIT STRUCTURE")
print("="*80)

circuit = """
INDIRECT OBJECT IDENTIFICATION (IOI) CIRCUIT IN GPT-2-SMALL

INPUT: "When [A] and [B] ... [B] gave ... to"
TARGET: Predict [A] (the indirect object)

LAYER GROUPS:

1. EARLY LAYERS (0-4): Previous Token Heads
   - L4H11: Strong previous-token attention (score: 1.0)
   - L2H2, L3H2: Moderate previous-token attention
   → Track sequential structure and position information

2. MIDDLE LAYERS (5-9): Name Mover Heads
   - L9H6: PRIMARY NAME MOVER (+1.98 contrib)
     * Attends strongly to first occurrence of IO name
     * Attention pattern: final → IO_first (~0.49 ABBA, ~0.34 BABA)
     * Moves IO information to final position
   
   - L9H5, L9H11: Supporting name movers
   - L5H1: Early name information gathering

3. LATE LAYERS (10-11): Output Heads

   A. S-INHIBITION / INVERSION HEADS (Positive contributors):
      - L11H0: +3.84 contribution (STRONGEST)
        * Attends to S_second position (0.23 in ABBA)
        * But promotes IO in output (inversion logic)
        * Implements "attend to subject, output the other name"
      
      - L11H8: +2.66 contribution
        * Attends to "gave" and comma (structural markers)
        * Helps identify the giving event
      
      - L11H5: +2.19 contribution
        * Attends to BOS/beginning tokens
        * Provides baseline context
      
      - L11H2: +1.75 contribution
        * Mixed attention pattern
   
   B. S-PROMOTION HEADS (Negative contributors - get overridden):
      - L8H3: -3.91 contribution
        * Attends to S_first (0.42 in ABBA)
        * Tries to promote S name
        * Gets suppressed by later positive heads
      
      - L10H2: -2.19 contribution
        * Similar S-promotion pattern

INFORMATION FLOW:

Token Embeddings
    ↓
[L0-4: Position tracking via previous-token heads]
    ↓
[L5-9: Name mover heads copy IO name to final position]
    ↓  (L9H6 is key: final token attends to IO_first)
    ↓
[L8-10: S-promotion heads try to output S]
    ↓  (L8H3, L10H2 attend to S and promote it)
    ↓
[L11: S-inhibition heads implement inversion]
    ↓  (L11H0, L11H8 attend to S but output IO)
    ↓
Final Logits → IO name preferred

KEY MECHANISMS:

1. NAME MOVEMENT:
   - L9H6 copies information from first IO mention to final position
   - Uses simple attention-based copying

2. SUBJECT IDENTIFICATION:
   - Model identifies S as the name that appears twice
   - S appears in middle position AND as subject of "gave"

3. INVERSION LOGIC:
   - Late layers (L11) implement "output the OTHER name"
   - Attend to subject positions but promote IO
   - This is the key insight: NOT just copying IO, but actively 
     suppressing S and inverting the attention pattern

4. COMPETITION:
   - Negative heads (L8H3) try to output S
   - Positive heads (L11H0, L11H8) override this
   - Net effect: IO wins by ~2-3 logits

QUANTITATIVE EVIDENCE:
- Effect size: ~3.5 logits (clean - corrupted)
- Works on both ABBA and BABA templates
- L9H6 attention to IO_first: 0.34-0.56 (very high)
- L11H0 contribution: +3.84 (largest single head)
- Sum of positive heads > sum of negative heads → IO predicted
"""

print(circuit)

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Define y-positions for different head types
layer_positions = {i: i for i in range(12)}

# Plot heads
positive_color = 'green'
negative_color = 'red'
neutral_color = 'gray'

# Plot positive heads
for layer, head, contrib in positive_heads[:8]:
    size = abs(contrib) * 100
    ax.scatter(head, layer, s=size, c=positive_color, alpha=0.7, edgecolors='black')
    ax.text(head + 0.3, layer, f'{contrib:.1f}', fontsize=8)

# Plot negative heads
for layer, head, contrib in negative_heads[:5]:
    size = abs(contrib) * 100
    ax.scatter(head, layer, s=size, c=negative_color, alpha=0.7, edgecolors='black')
    ax.text(head + 0.3, layer, f'{contrib:.1f}', fontsize=8)

# Add labels
ax.set_xlabel('Head', fontsize=12)
ax.set_ylabel('Layer', fontsize=12)
ax.set_title('IOI Circuit: Attention Head Contributions\n(size = |contribution|, green = positive, red = negative)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(-0.5, 11.5)
ax.set_xlim(-0.5, 11.5)
ax.grid(True, alpha=0.3)

# Add legend
green_patch = mpatches.Patch(color=positive_color, label='Promotes IO')
red_patch = mpatches.Patch(color=negative_color, label='Promotes S')
ax.legend(handles=[green_patch, red_patch], loc='upper right')

# Add key head annotations
annotations = [
    (9, 6, 'Name Mover'),
    (11, 0, 'S-Inhibition'),
    (11, 8, 'Structural'),
    (8, 3, 'S-Promotion'),
]

for layer, head, label in annotations:
    ax.annotate(label, xy=(head, layer), xytext=(head+1, layer+0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('scratch/ioi_circuit_diagram.png', dpi=150)
print("\nSaved circuit diagram to scratch/ioi_circuit_diagram.png")

# Save full analysis
analysis = {
    'phenomenon': 'Indirect Object Identification',
    'effect_size': '~3.5 logits',
    'key_heads': {
        'name_mover': [(9, 6, 1.98)],
        's_inhibition': [(11, 0, 3.84), (11, 8, 2.66), (11, 5, 2.19), (11, 2, 1.75)],
        's_promotion': [(8, 3, -3.91), (10, 2, -2.19)]
    },
    'mechanism': [
        '1. Previous token heads (L2-4) track sequential structure',
        '2. Name mover heads (L9H6) copy IO name to final position',
        '3. S-promotion heads (L8H3) try to output S',
        '4. S-inhibition heads (L11H0, L11H8) implement inversion logic',
        '5. Final output: IO name preferred over S name'
    ]
}

with open('scratch/ioi_circuit_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("\nSaved complete analysis to scratch/ioi_circuit_analysis.json")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
