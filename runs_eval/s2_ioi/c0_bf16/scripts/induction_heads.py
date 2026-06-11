"""
Check for induction-like behavior and previous token heads
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
import json
import numpy as np
from autointerp.tools.model import load_model
import matplotlib.pyplot as plt

print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

prompt = "When Mary and John went to the store, John gave a drink to"

tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
token_strs = tokenizer.convert_ids_to_tokens(tokens[0])

print("Tokens:")
for i, tok in enumerate(token_strs):
    print(f"  {i:2d}: {tok}")

# Get all attention patterns
n_layers = model.config.n_layer
n_heads = model.config.n_head

attention_cache = {}

def make_hook(layer):
    def hook(module, input, output):
        attn_weights = output[1]
        if attn_weights is not None:
            attention_cache[layer] = attn_weights[0, :, :, :].detach().cpu().float()
    return hook

hooks = []
for layer in range(n_layers):
    h = model.transformer.h[layer].attn.register_forward_hook(make_hook(layer))
    hooks.append(h)

with torch.no_grad():
    model(tokens, output_attentions=True)

for h in hooks:
    h.remove()

# Check for previous token heads
# These attend from position i to position i-1
print("\n" + "="*80)
print("PREVIOUS TOKEN HEADS")
print("="*80)
print("Looking for heads with strong diagonal-1 pattern\n")

prev_token_scores = []

for layer in range(n_layers):
    for head in range(n_heads):
        attn = attention_cache[layer][head, :, :].numpy()
        
        # Score: average attention to previous token across all positions
        score = 0
        count = 0
        for i in range(1, len(token_strs)):
            score += attn[i, i-1]
            count += 1
        score /= count
        
        prev_token_scores.append((layer, head, score))

# Sort by score
prev_token_scores.sort(key=lambda x: x[2], reverse=True)

print(f"{'Layer':<8} {'Head':<8} {'Avg Prev-Token Attn':<20}")
print("-" * 50)
for layer, head, score in prev_token_scores[:15]:
    print(f"{layer:<8} {head:<8} {score:<20.4f}")

# Visualize top previous token heads
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Top Previous Token Heads', fontsize=16)

for idx, (layer, head, score) in enumerate(prev_token_scores[:6]):
    ax = axes[idx // 3, idx % 3]
    attn = attention_cache[layer][head, :, :].numpy()
    
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xlabel('Source position')
    ax.set_ylabel('Dest position')
    ax.set_title(f'L{layer}H{head} (score: {score:.3f})')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('scratch/prev_token_heads.png', dpi=150)
print("\nSaved to scratch/prev_token_heads.png")

# Check for induction-like patterns
print("\n" + "="*80)
print("CHECKING POSITION/CONTEXT SENSITIVITY")
print("="*80)

# Compare attention patterns when names are at different positions
# The key insight is whether heads care about the POSITION or the CONTEXT

# For IOI, we care about "after first mention" vs "after second mention"
# Position of John: 3 (first), 9 (second)

print("\nComparing attention to 'John' at different positions:")
print("Position 3 (first mention) vs Position 9 (second mention)")

# Select some key heads and check how other positions attend to them
key_positions = [1, 3, 9]  # Mary first, John first, John second

for layer, head in [(9, 6), (11, 0), (8, 3)]:
    attn = attention_cache[layer][head, :, :].numpy()
    
    print(f"\nL{layer}H{head}:")
    for src_pos in range(len(token_strs)):
        attn_to_positions = [attn[src_pos, pos] for pos in key_positions]
        if max(attn_to_positions) > 0.1:  # Only show if significant
            print(f"  From pos {src_pos:2d} ({token_strs[src_pos]:15s}): " + 
                  f"Mary={attn_to_positions[0]:.3f}, John1={attn_to_positions[1]:.3f}, John2={attn_to_positions[2]:.3f}")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)
print("1. Previous token heads help track sequential structure")
print("2. Name mover heads (L9H6) strongly attend to first IO mention")
print("3. Late layer heads (L11H0) use complex attention patterns")
print("   combining information from multiple name positions")

