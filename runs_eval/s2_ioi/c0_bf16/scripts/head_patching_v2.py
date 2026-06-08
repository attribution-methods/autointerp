"""
Patch individual attention heads using TransformerLens-style approach
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
import json
import numpy as np
from autointerp.tools.model import load_model
from tqdm import tqdm

# Load model
print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

# Use ABBA example
clean_prompt = "When Mary and John went to the store, John gave a drink to"
corrupt_prompt = "When Mary and John went to the store, Mary gave a drink to"
io_name = "Mary"
s_name = "John"

# Get tokens
io_token = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
s_token = tokenizer.encode(" " + s_name, add_special_tokens=False)[0]

clean_tokens = tokenizer.encode(clean_prompt, return_tensors="pt").to(handle.device)
corrupt_tokens = tokenizer.encode(corrupt_prompt, return_tensors="pt").to(handle.device)

print(f"Tokens: {tokenizer.convert_ids_to_tokens(clean_tokens[0])}")
print(f"Sequence length: {clean_tokens.shape[1]}")

def get_logit_diff(tokens):
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, -1, :]
    return logits[io_token].item() - logits[s_token].item()

# Get baselines
clean_diff = get_logit_diff(clean_tokens)
corrupt_diff = get_logit_diff(corrupt_tokens)
effect_size = clean_diff - corrupt_diff

print(f"\nClean logit diff: {clean_diff:.3f}")
print(f"Corrupt logit diff: {corrupt_diff:.3f}")
print(f"Effect size: {effect_size:.3f}")

# Use helper tools for patching
from autointerp.tools.patching import sweep_patch_sites

n_layers = model.config.n_layer
n_heads = model.config.n_head

# Define patch sites: attention output for each head at final token position
sites = []
final_pos = clean_tokens.shape[1] - 1
for layer in range(n_layers):
    for head in range(n_heads):
        sites.append({
            'layer': layer,
            'component': 'attn',
            'head': head,
            'position': final_pos
        })

print(f"\nPatching {len(sites)} attention heads at position {final_pos}...")

# Use sweep_patch_sites helper
try:
    results = sweep_patch_sites(
        handle,
        clean=clean_prompt,
        corrupt=corrupt_prompt,
        sites=sites
    )
    print("Sweep patching completed!")
    print(f"Results keys: {results.keys()}")
except Exception as e:
    print(f"Error with sweep_patch_sites: {e}")
    print("Falling back to manual patching...")
    
    # Manual patching approach
    head_results = []
    
    # First, run clean and corrupt to get activation cache
    def get_head_activations(tokens):
        """Get attention head outputs for all layers"""
        activations = {}
        
        def make_hook(layer):
            def hook(module, input, output):
                # output is (attn_output, attn_weights)
                attn_out = output[0]  # shape: [batch, seq, hidden]
                batch, seq, hidden = attn_out.shape
                d_head = hidden // n_heads
                # Reshape to [batch, seq, n_heads, d_head]
                attn_out_heads = attn_out.reshape(batch, seq, n_heads, d_head)
                activations[layer] = attn_out_heads.detach().clone()
            return hook
        
        hooks = []
        for layer in range(n_layers):
            hook = model.transformer.h[layer].attn.register_forward_hook(make_hook(layer))
            hooks.append(hook)
        
        with torch.no_grad():
            model(tokens)
        
        for hook in hooks:
            hook.remove()
        
        return activations
    
    print("Caching clean activations...")
    clean_acts = get_head_activations(clean_tokens)
    
    print("Patching individual heads...")
    for layer in tqdm(range(n_layers)):
        for head in range(n_heads):
            # Patch this specific head
            def patch_hook(module, input, output):
                attn_out = output[0]
                batch, seq, hidden = attn_out.shape
                d_head = hidden // n_heads
                attn_out_heads = attn_out.reshape(batch, seq, n_heads, d_head)
                # Replace this head's output with clean version
                attn_out_heads[:, :, head, :] = clean_acts[layer][:, :, head, :]
                attn_out = attn_out_heads.reshape(batch, seq, hidden)
                return (attn_out,) + output[1:]
            
            hook = model.transformer.h[layer].attn.register_forward_hook(patch_hook)
            patched_diff = get_logit_diff(corrupt_tokens)
            hook.remove()
            
            recovery = (patched_diff - corrupt_diff) / effect_size if effect_size != 0 else 0
            
            head_results.append({
                'layer': layer,
                'head': head,
                'patched_diff': patched_diff,
                'recovery': recovery
            })
    
    # Sort and display
    head_results.sort(key=lambda x: x['recovery'], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT ATTENTION HEADS")
    print("="*80)
    print(f"{'Layer':<8} {'Head':<8} {'Recovery %':<15}")
    print("-" * 80)
    for r in head_results[:20]:
        print(f"{r['layer']:<8} {r['head']:<8} {r['recovery']*100:<15.1f}")
    
    # Save
    with open('scratch/head_patching_results.json', 'w') as f:
        json.dump({
            'clean_diff': clean_diff,
            'corrupt_diff': corrupt_diff,
            'effect_size': effect_size,
            'head_results': head_results
        }, f, indent=2)
    
    print("\nSaved to scratch/head_patching_results.json")
