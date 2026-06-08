"""
Detailed analysis of attention patterns in key heads.
Focus on L9H9, L10H6, L11H10 and others that show strong differential attention.
"""
import sys
sys.path.insert(0, '/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp')

import torch
import numpy as np
from autointerp.tools.model import load_model
import json

# Load GPT-2-small
print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

# Test on multiple examples
test_cases = [
    {
        'clean': "When John and Mary went to the store, Mary gave a drink to",
        'corrupt': "When Mary and John went to the store, Mary gave a drink to",
        'name_a': "John",
        'name_b': "Mary"
    },
    {
        'clean': "When Alice and Bob went to the beach, Bob gave a gift to",
        'corrupt': "When Bob and Alice went to the beach, Bob gave a gift to",
        'name_a': "Alice",
        'name_b': "Bob"
    },
]

print("\n" + "="*80)
print("ATTENTION PATTERN ANALYSIS")
print("="*80)

key_heads = [
    (9, 9),   # Strong differential attention
    (10, 6),  # Strong differential attention
    (11, 10), # Shows pattern shift
]

for test_idx, test_case in enumerate(test_cases):
    print(f"\n{'='*80}")
    print(f"TEST CASE {test_idx + 1}")
    print(f"{'='*80}")
    
    clean_prompt = test_case['clean']
    corrupt_prompt = test_case['corrupt']
    name_a = test_case['name_a']
    name_b = test_case['name_b']
    
    print(f"Clean:   {clean_prompt}")
    print(f"Corrupt: {corrupt_prompt}")
    
    name_a_id = tokenizer.encode(" " + name_a, add_special_tokens=False)[0]
    name_b_id = tokenizer.encode(" " + name_b, add_special_tokens=False)[0]
    
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(handle.device)
    corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(handle.device)
    
    # Get token strings for visualization
    clean_token_strs = [tokenizer.decode([t]) for t in clean_tokens[0]]
    
    # Identify key positions
    clean_token_list = clean_tokens[0].cpu().tolist()
    
    # Find positions of key tokens
    name_a_positions = [i for i, t in enumerate(clean_token_list) if t == name_a_id]
    name_b_positions = [i for i, t in enumerate(clean_token_list) if t == name_b_id]
    
    # Find S2 position (second occurrence of subject name B)
    s2_pos = name_b_positions[-1] if len(name_b_positions) > 1 else None
    
    # Find IO position (first occurrence of indirect object name A)
    io_pos = name_a_positions[0] if name_a_positions else None
    
    print(f"\nKey positions:")
    print(f"  IO ('{name_a}'): {io_pos}")
    print(f"  S2 ('{name_b}'): {s2_pos}")
    print(f"  Final position: {len(clean_token_list) - 1}")
    
    with torch.no_grad():
        clean_output = model(clean_tokens, output_attentions=True)
        corrupt_output = model(corrupt_tokens, output_attentions=True)
    
    clean_attentions = clean_output.attentions
    corrupt_attentions = corrupt_output.attentions
    
    for layer, head in key_heads:
        print(f"\n  Layer {layer}, Head {head}:")
        
        clean_attn = clean_attentions[layer][0, head, -1, :]  # Attention from final position
        corrupt_attn = corrupt_attentions[layer][0, head, -1, :]
        
        # Show attention to key positions
        print(f"    Clean attention pattern (from final token):")
        for i, (token_str, attn_val) in enumerate(zip(clean_token_strs, clean_attn)):
            if attn_val > 0.05 or i in [io_pos, s2_pos]:
                marker = ""
                if i == io_pos:
                    marker = " <-- IO"
                elif i == s2_pos:
                    marker = " <-- S2"
                print(f"      [{i:2d}] '{token_str:10s}': {attn_val.item():.3f}{marker}")
        
        if io_pos is not None and s2_pos is not None:
            clean_attn_io = clean_attn[io_pos].item()
            clean_attn_s2 = clean_attn[s2_pos].item()
            corrupt_attn_io = corrupt_attn[io_pos].item()
            corrupt_attn_s2 = corrupt_attn[s2_pos].item()
            
            print(f"    Summary:")
            print(f"      Clean:   Attn(IO)={clean_attn_io:.3f}, Attn(S2)={clean_attn_s2:.3f}, Diff={clean_attn_io - clean_attn_s2:.3f}")
            print(f"      Corrupt: Attn(IO)={corrupt_attn_io:.3f}, Attn(S2)={corrupt_attn_s2:.3f}, Diff={corrupt_attn_io - corrupt_attn_s2:.3f}")

print("\n" + "="*80)
print("ANALYZING OTHER LAYERS FOR DUPLICATE TOKEN DETECTION")
print("="*80)

# Look at earlier layers to see if they detect duplicates
for test_idx, test_case in enumerate(test_cases):
    if test_idx > 0:
        break  # Just analyze first case in detail
    
    clean_prompt = test_case['clean']
    name_a = test_case['name_a']
    name_b = test_case['name_b']
    
    print(f"\nPrompt: {clean_prompt}")
    
    name_a_id = tokenizer.encode(" " + name_a, add_special_tokens=False)[0]
    name_b_id = tokenizer.encode(" " + name_b, add_special_tokens=False)[0]
    
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(handle.device)
    clean_token_list = clean_tokens[0].cpu().tolist()
    clean_token_strs = [tokenizer.decode([t]) for t in clean_tokens[0]]
    
    name_b_positions = [i for i, t in enumerate(clean_token_list) if t == name_b_id]
    s2_pos = name_b_positions[-1] if len(name_b_positions) > 1 else None
    s1_pos = name_b_positions[0] if len(name_b_positions) > 1 else None
    
    if s1_pos is not None and s2_pos is not None:
        print(f"S1 position (first '{name_b}'): {s1_pos}")
        print(f"S2 position (second '{name_b}'): {s2_pos}")
        
        with torch.no_grad():
            clean_output = model(clean_tokens, output_attentions=True)
        
        clean_attentions = clean_output.attentions
        
        # Check if S2 attends to S1 (duplicate token detection)
        print(f"\nAttention from S2 to S1 across layers:")
        for layer in range(model.config.n_layer):
            for head in range(model.config.n_head):
                attn_s2_to_s1 = clean_attentions[layer][0, head, s2_pos, s1_pos].item()
                if attn_s2_to_s1 > 0.3:  # Significant attention
                    print(f"  L{layer:2d}H{head:2d}: {attn_s2_to_s1:.3f}")

print("\n" + "="*80)
print("CHECKING FOR S-INHIBITION HEADS")
print("="*80)

# S-inhibition heads: attend to S2 and suppress it in the output
# Look at earlier layers that might be suppressing the repeated name

for test_idx, test_case in enumerate(test_cases):
    if test_idx > 0:
        break
    
    clean_prompt = test_case['clean']
    name_b = test_case['name_b']
    
    name_b_id = tokenizer.encode(" " + name_b, add_special_tokens=False)[0]
    
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(handle.device)
    clean_token_list = clean_tokens[0].cpu().tolist()
    
    name_b_positions = [i for i, t in enumerate(clean_token_list) if t == name_b_id]
    s2_pos = name_b_positions[-1] if len(name_b_positions) > 1 else None
    
    if s2_pos is not None:
        with torch.no_grad():
            clean_output = model(clean_tokens, output_attentions=True)
        
        clean_attentions = clean_output.attentions
        
        print(f"\nHeads with strong attention to S2 at final position:")
        for layer in range(model.config.n_layer):
            for head in range(model.config.n_head):
                attn_final_to_s2 = clean_attentions[layer][0, head, -1, s2_pos].item()
                if attn_final_to_s2 > 0.2:
                    print(f"  L{layer:2d}H{head:2d}: {attn_final_to_s2:.3f}")

print("\nAnalysis complete!")
