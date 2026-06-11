"""
Patch individual attention heads to identify important ones for IOI
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
import json
import numpy as np
from autointerp.tools.model import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt

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

print(f"\nClean prompt:   {clean_prompt}")
print(f"Corrupt prompt: {corrupt_prompt}")

# Get tokens
io_token = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
s_token = tokenizer.encode(" " + s_name, add_special_tokens=False)[0]

def get_logit_diff(tokens):
    """Get the logit difference (IO - S) for tokenized input"""
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, -1, :]
    return logits[io_token].item() - logits[s_token].item()

# Tokenize
clean_tokens = tokenizer.encode(clean_prompt, return_tensors="pt").to(handle.device)
corrupt_tokens = tokenizer.encode(corrupt_prompt, return_tensors="pt").to(handle.device)

print(f"\nTokens: {tokenizer.convert_ids_to_tokens(clean_tokens[0])}")

# Get baselines
clean_diff = get_logit_diff(clean_tokens)
corrupt_diff = get_logit_diff(corrupt_tokens)
effect_size = clean_diff - corrupt_diff

print(f"\nClean logit diff: {clean_diff:.3f}")
print(f"Corrupt logit diff: {corrupt_diff:.3f}")
print(f"Effect size: {effect_size:.3f}")

# Cache attention head outputs for both runs
n_layers = model.config.n_layer
n_heads = model.config.n_head

def cache_head_outputs(tokens):
    """Cache the output of each attention head"""
    cache = {}
    
    def make_hook(layer, head):
        def hook(module, input, output):
            # output[0] is attention output, shape [batch, seq, n_heads, d_head]
            attn_out = output[0]
            # Split by head and store
            batch, seq, hidden = attn_out.shape
            d_head = hidden // n_heads
            attn_out_by_head = attn_out.reshape(batch, seq, n_heads, d_head)
            cache[(layer, head)] = attn_out_by_head[:, :, head, :].detach()
        return hook
    
    hooks = []
    for layer in range(n_layers):
        hook = model.transformer.h[layer].attn.register_forward_hook(make_hook(layer, 0))
        hooks.append(hook)
    
    with torch.no_grad():
        model(tokens)
    
    for hook in hooks:
        hook.remove()
    
    return cache

print("\nCaching attention head outputs...")
clean_cache = cache_head_outputs(clean_tokens)
corrupt_cache = cache_head_outputs(corrupt_tokens)

# Patch each attention head
print("\nPatching individual attention heads...")
head_results = []

for layer in tqdm(range(n_layers)):
    for head in range(n_heads):
        # Patch this head: run on corrupt input but replace this head's output with clean
        patched_value = None
        
        def patch_hook(module, input, output):
            # Get attention output
            attn_out = output[0]
            batch, seq, hidden = attn_out.shape
            d_head = hidden // n_heads
            
            # Reshape to separate heads
            attn_out_by_head = attn_out.reshape(batch, seq, n_heads, d_head)
            
            # Replace the specific head with clean version
            attn_out_by_head[:, :, head, :] = clean_cache[(layer, head)]
            
            # Reshape back
            attn_out = attn_out_by_head.reshape(batch, seq, hidden)
            
            return (attn_out,) + output[1:]
        
        hook = model.transformer.h[layer].attn.register_forward_hook(patch_hook)
        patched_diff = get_logit_diff(corrupt_tokens)
        hook.remove()
        
        # Calculate recovery
        recovery = (patched_diff - corrupt_diff) / effect_size if effect_size != 0 else 0
        
        head_results.append({
            'layer': layer,
            'head': head,
            'patched_diff': patched_diff,
            'recovery': recovery
        })

# Sort by recovery
head_results.sort(key=lambda x: x['recovery'], reverse=True)

print("\n" + "="*80)
print("TOP 20 MOST IMPORTANT ATTENTION HEADS (by recovery)")
print("="*80)
print(f"{'Layer':<8} {'Head':<8} {'Recovery %':<15} {'Patched Diff':<15}")
print("-" * 80)
for r in head_results[:20]:
    print(f"{r['layer']:<8} {r['head']:<8} {r['recovery']*100:<15.1f} {r['patched_diff']:<15.3f}")

# Save results
with open('scratch/head_patching_results.json', 'w') as f:
    json.dump({
        'clean_diff': clean_diff,
        'corrupt_diff': corrupt_diff,
        'effect_size': effect_size,
        'head_results': head_results
    }, f, indent=2)

print("\nSaved to scratch/head_patching_results.json")

# Create heatmap
recovery_matrix = np.zeros((n_layers, n_heads))
for r in head_results:
    recovery_matrix[r['layer'], r['head']] = r['recovery']

plt.figure(figsize=(12, 8))
plt.imshow(recovery_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=1.0)
plt.colorbar(label='Recovery (fraction of effect restored)')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.title('Attention Head Importance for IOI (Activation Patching)')
for i in range(n_layers):
    for j in range(n_heads):
        if recovery_matrix[i, j] > 0.3:
            plt.text(j, i, f'{recovery_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('scratch/head_patching_heatmap.png', dpi=150)
print("Saved heatmap to scratch/head_patching_heatmap.png")
