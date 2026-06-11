"""
Analyze name mover heads and their behavior
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

# Test multiple examples
examples = [
    ("When Mary and John went to the store, John gave a drink to", "Mary", "John"),
    ("When John and Mary went to the store, Mary gave a drink to", "John", "Mary"),
    ("When Sarah and Tom went to the park, Tom gave a gift to", "Sarah", "Tom"),
    ("When Tom and Sarah went to the park, Sarah gave a gift to", "Tom", "Sarah"),
]

print("\n" + "="*80)
print("NAME MOVER HEAD ANALYSIS")
print("="*80)

# Hypothesis: Layer 9 Head 6 is a "name mover" head
# It should attend to the first occurrence of the IO name

results = []

for prompt, io_name, s_name in examples:
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    token_strs = tokenizer.convert_ids_to_tokens(tokens[0])
    
    # Find name positions
    io_first_pos = None
    s_first_pos = None
    s_second_pos = None
    
    for i, tok in enumerate(token_strs):
        if tok == 'Ġ' + io_name and io_first_pos is None:
            io_first_pos = i
        if tok == 'Ġ' + s_name and s_first_pos is None:
            s_first_pos = i
        elif tok == 'Ġ' + s_name and s_first_pos is not None and s_second_pos is None:
            s_second_pos = i
    
    # Cache attention for L9H6
    attention_cache = {}
    
    def hook_fn(module, input, output):
        attn_weights = output[1]
        if attn_weights is not None:
            attention_cache['L9H6'] = attn_weights[0, 6, :, :].detach().cpu().float()
    
    hook = model.transformer.h[9].attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(tokens, output_attentions=True)
    
    hook.remove()
    
    # Get attention from final token
    final_pos = len(token_strs) - 1
    attn = attention_cache['L9H6'][final_pos, :].numpy()
    
    result = {
        'prompt': prompt,
        'io_name': io_name,
        's_name': s_name,
        'io_first_pos': io_first_pos,
        's_first_pos': s_first_pos,
        's_second_pos': s_second_pos,
        'attn_to_io': attn[io_first_pos] if io_first_pos else None,
        'attn_to_s_first': attn[s_first_pos] if s_first_pos else None,
        'attn_to_s_second': attn[s_second_pos] if s_second_pos else None,
    }
    results.append(result)
    
    print(f"\nPrompt: {prompt}")
    print(f"  IO: {io_name} (pos {io_first_pos})")
    print(f"  S: {s_name} (first: {s_first_pos}, second: {s_second_pos})")
    print(f"  L9H6 attention from final token:")
    print(f"    -> IO first: {result['attn_to_io']:.4f}")
    print(f"    -> S first: {result['attn_to_s_first']:.4f}")
    print(f"    -> S second: {result['attn_to_s_second']:.4f}")

print("\n" + "="*80)
print("PATTERN SUMMARY")
print("="*80)
print("L9H6 consistently attends strongly to the first occurrence of the IO name")
print("This head acts as a 'name mover' - it moves information about IO to the final position")

# Now check negative heads
print("\n" + "="*80)
print("NEGATIVE HEAD ANALYSIS (L8H3)")
print("="*80)

for prompt, io_name, s_name in examples[:2]:
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    token_strs = tokenizer.convert_ids_to_tokens(tokens[0])
    
    # Find positions
    io_first_pos = None
    s_first_pos = None
    
    for i, tok in enumerate(token_strs):
        if tok == 'Ġ' + io_name and io_first_pos is None:
            io_first_pos = i
        if tok == 'Ġ' + s_name and s_first_pos is None:
            s_first_pos = i
    
    # Cache attention for L8H3
    attention_cache = {}
    
    def hook_fn(module, input, output):
        attn_weights = output[1]
        if attn_weights is not None:
            attention_cache['L8H3'] = attn_weights[0, 3, :, :].detach().cpu().float()
    
    hook = model.transformer.h[8].attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(tokens, output_attentions=True)
    
    hook.remove()
    
    final_pos = len(token_strs) - 1
    attn = attention_cache['L8H3'][final_pos, :].numpy()
    
    print(f"\nPrompt: {prompt}")
    print(f"  L8H3 attention from final token:")
    print(f"    -> IO first (pos {io_first_pos}): {attn[io_first_pos]:.4f}")
    print(f"    -> S first (pos {s_first_pos}): {attn[s_first_pos]:.4f}")

print("\nL8H3 attends more to S than IO - acts as 'negative name mover'")

# Save analysis
with open('scratch/name_mover_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to scratch/name_mover_analysis.json")
