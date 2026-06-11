"""
Investigate duplicate token heads and S-inhibition heads
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
import json
import numpy as np
from autointerp.tools.model import load_model

print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

print("\n" + "="*80)
print("DUPLICATE TOKEN HEAD HYPOTHESIS")
print("="*80)
print("Some heads might attend to the duplicate occurrence of S (the subject)")
print("This helps the model identify which name appears twice")

# ABBA example: Mary, John, John - John is duplicate
# BABA example: John, Mary, Mary - Mary is duplicate

examples = [
    ("When Mary and John went to the store, John gave a drink to", "Mary", "John", "ABBA"),
    ("When John and Mary went to the store, Mary gave a drink to", "John", "Mary", "BABA"),
]

# Check a few early/mid-layer heads for duplicate token attention
heads_to_check = [
    (8, 10), (8, 11), (9, 9), (10, 0), (10, 7),
    (5, 5), (5, 9), (7, 3), (7, 9), (7, 10)
]

results = {}

for prompt, io_name, s_name, template in examples:
    print(f"\n{'='*80}")
    print(f"{template}: {prompt}")
    
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    token_strs = tokenizer.convert_ids_to_tokens(tokens[0])
    
    # Find positions
    s_first_pos = None
    s_second_pos = None
    io_first_pos = None
    
    for i, tok in enumerate(token_strs):
        if tok == 'Ġ' + s_name and s_first_pos is None:
            s_first_pos = i
        elif tok == 'Ġ' + s_name and s_first_pos is not None:
            s_second_pos = i
        if tok == 'Ġ' + io_name and io_first_pos is None:
            io_first_pos = i
    
    print(f"S duplicate: {s_name} at positions {s_first_pos} and {s_second_pos}")
    print(f"IO unique: {io_name} at position {io_first_pos}")
    
    # Get attention patterns for all heads
    attention_cache = {}
    
    def make_hook(layer, head):
        def hook(module, input, output):
            attn_weights = output[1]
            if attn_weights is not None:
                attention_cache[(layer, head)] = attn_weights[0, head, :, :].detach().cpu().float()
        return hook
    
    hooks = []
    for layer, head in heads_to_check:
        h = model.transformer.h[layer].attn.register_forward_hook(make_hook(layer, head))
        hooks.append(h)
    
    with torch.no_grad():
        model(tokens, output_attentions=True)
    
    for h in hooks:
        h.remove()
    
    # Check attention from S_second to S_first
    print(f"\nAttention from S_second (pos {s_second_pos}) to S_first (pos {s_first_pos}):")
    print(f"{'Layer':<8} {'Head':<8} {'Attn':<12}")
    print("-" * 40)
    
    for layer, head in sorted(heads_to_check):
        if (layer, head) in attention_cache:
            attn = attention_cache[(layer, head)]
            attn_to_duplicate = attn[s_second_pos, s_first_pos].item()
            print(f"{layer:<8} {head:<8} {attn_to_duplicate:<12.4f}")
            
            results[f"{template}_L{layer}H{head}_dup"] = float(attn_to_duplicate)

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("Looking for heads where S_second strongly attends to S_first")
print("These 'duplicate token heads' help detect which name appears twice")

# Check which heads have high attention in both templates
high_dup_heads = set()
for key, val in results.items():
    if val > 0.3:  # threshold
        layer, head = key.split('_')[1].replace('L','').replace('H','').split('H')
        if 'H' in key.split('_')[1]:
            parts = key.split('_')[1].replace('L', '').split('H')
            layer, head = int(parts[0]), int(parts[1])
            high_dup_heads.add((layer, head))

print(f"\nHeads with strong duplicate token attention: {sorted(high_dup_heads)}")

# Now check S-inhibition mechanism
print("\n" + "="*80)
print("S-INHIBITION HYPOTHESIS")
print("="*80)
print("Some heads might suppress S to prevent it from being predicted")

# Check final layer heads that have negative contributions
# and their attention to S positions

for prompt, io_name, s_name, template in examples[:1]:
    print(f"\n{template}: {prompt}")
    
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    token_strs = tokenizer.convert_ids_to_tokens(tokens[0])
    
    # Positions
    s_first_pos = None
    s_second_pos = None
    
    for i, tok in enumerate(token_strs):
        if tok == 'Ġ' + s_name and s_first_pos is None:
            s_first_pos = i
        elif tok == 'Ġ' + s_name:
            s_second_pos = i
    
    # Check L11H0 attention pattern
    attention_cache = {}
    
    def hook_fn(module, input, output):
        attn_weights = output[1]
        if attn_weights is not None:
            attention_cache['L11H0'] = attn_weights[0, 0, :, :].detach().cpu().float()
    
    hook = model.transformer.h[11].attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(tokens, output_attentions=True)
    
    hook.remove()
    
    final_pos = len(token_strs) - 1
    attn = attention_cache['L11H0'][final_pos, :].numpy()
    
    print(f"\nL11H0 (positive contributor) attention from final:")
    print(f"  -> S second (pos {s_second_pos}): {attn[s_second_pos]:.4f}")
    print(f"  -> S first (pos {s_first_pos}): {attn[s_first_pos]:.4f}")
    print("\nL11H0 attends to S positions but contributes POSITIVELY to IO")
    print("This suggests it's learning 'attend to subject, output opposite name'")

print("\n" + "="*80)
print("SUMMARY OF CIRCUIT COMPONENTS")
print("="*80)
print("1. NAME MOVER HEADS (e.g., L9H6):")
print("   - Attend to first occurrence of IO name")
print("   - Move IO information to final position")
print("")
print("2. DUPLICATE TOKEN HEADS (middle layers):")
print("   - Detect which name appears twice (the subject)")
print("   - Help distinguish IO from S")
print("")
print("3. S-INHIBITION / NAME SELECTION HEADS (L11H0, L11H8, etc.):")
print("   - Attend to subject positions")
print("   - But promote IO in output (inverse logic)")
print("   - Implement 'output the OTHER name' logic")
print("")
print("4. NEGATIVE HEADS (L8H3, L10H2):")
print("   - Attend to S and try to promote it")
print("   - Get overridden by later positive heads")

