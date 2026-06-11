"""
Activation patching to identify critical layers and components for IOI.
Custom implementation to patch activations from corrupt to clean.
"""
import sys
sys.path.insert(0, '/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp')

import torch
import numpy as np
from autointerp.tools.model import load_model
import json
from functools import partial

# Load GPT-2-small
print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

# Use a clean and corrupted version of the same template
clean_prompt = "When John and Mary went to the store, Mary gave a drink to"
corrupt_prompt = "When Mary and John went to the store, Mary gave a drink to"

# The target names
name_a = "John"
name_b = "Mary"

# Get token IDs
name_a_id = tokenizer.encode(" " + name_a, add_special_tokens=False)[0]
name_b_id = tokenizer.encode(" " + name_b, add_special_tokens=False)[0]

def compute_logit_diff(logits):
    """Compute logit difference between name A and name B."""
    return logits[name_a_id].item() - logits[name_b_id].item()

print("\n" + "="*80)
print("BASELINE LOGIT DIFFERENCES")
print("="*80)

clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(handle.device)
corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(handle.device)

# Get clean activations
clean_cache = {}
def save_clean_activation(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            clean_cache[name] = output[0].detach().clone()
        else:
            clean_cache[name] = output.detach().clone()
    return hook

# Register hooks to save clean activations
hooks = []
n_layers = model.config.n_layer

for layer_idx in range(n_layers):
    layer = model.transformer.h[layer_idx]
    # Attention output
    hooks.append(layer.attn.register_forward_hook(save_clean_activation(f'attn_{layer_idx}')))
    # MLP output
    hooks.append(layer.mlp.register_forward_hook(save_clean_activation(f'mlp_{layer_idx}')))

with torch.no_grad():
    clean_logits = model(clean_tokens).logits[0, -1, :]
    
# Remove hooks
for hook in hooks:
    hook.remove()

clean_logit_diff = compute_logit_diff(clean_logits)
print(f"Clean prompt: {clean_prompt}")
print(f"Clean logit diff (A - B): {clean_logit_diff:.3f}")

# Get corrupt baseline
with torch.no_grad():
    corrupt_logits = model(corrupt_tokens).logits[0, -1, :]

corrupt_logit_diff = compute_logit_diff(corrupt_logits)
print(f"\nCorrupt prompt: {corrupt_prompt}")
print(f"Corrupt logit diff (A - B): {corrupt_logit_diff:.3f}")
print(f"\nDifference: {clean_logit_diff - corrupt_logit_diff:.3f}")

print("\n" + "="*80)
print("ACTIVATION PATCHING SWEEP")
print("="*80)

patching_results = []

# Patch attention outputs
print("\nPatching attention outputs...")
for layer_idx in range(n_layers):
    def patch_hook(module, input, output, layer_idx=layer_idx):
        # Replace the corrupt activation with clean activation
        if isinstance(output, tuple):
            # output is (hidden_states, ...)
            patched = clean_cache[f'attn_{layer_idx}'].clone()
            return (patched,) + output[1:]
        else:
            return clean_cache[f'attn_{layer_idx}'].clone()
    
    layer = model.transformer.h[layer_idx]
    hook = layer.attn.register_forward_hook(patch_hook)
    
    with torch.no_grad():
        patched_logits = model(corrupt_tokens).logits[0, -1, :]
    
    hook.remove()
    
    patched_logit_diff = compute_logit_diff(patched_logits)
    recovery = (patched_logit_diff - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)
    
    patching_results.append({
        'layer': layer_idx,
        'component': 'attn',
        'patched_logit_diff': patched_logit_diff,
        'recovery': recovery
    })
    
    if recovery > 0.1:
        print(f"Layer {layer_idx:2d}, attn : Recovery = {recovery:6.1%}, Logit diff = {patched_logit_diff:.3f}")

# Patch MLP outputs
print("\nPatching MLP outputs...")
for layer_idx in range(n_layers):
    def patch_hook(module, input, output, layer_idx=layer_idx):
        return clean_cache[f'mlp_{layer_idx}'].clone()
    
    layer = model.transformer.h[layer_idx]
    hook = layer.mlp.register_forward_hook(patch_hook)
    
    with torch.no_grad():
        patched_logits = model(corrupt_tokens).logits[0, -1, :]
    
    hook.remove()
    
    patched_logit_diff = compute_logit_diff(patched_logits)
    recovery = (patched_logit_diff - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)
    
    patching_results.append({
        'layer': layer_idx,
        'component': 'mlp',
        'patched_logit_diff': patched_logit_diff,
        'recovery': recovery
    })
    
    if recovery > 0.1:
        print(f"Layer {layer_idx:2d}, mlp  : Recovery = {recovery:6.1%}, Logit diff = {patched_logit_diff:.3f}")

# Save results
output_data = {
    'clean_logit_diff': clean_logit_diff,
    'corrupt_logit_diff': corrupt_logit_diff,
    'clean_prompt': clean_prompt,
    'corrupt_prompt': corrupt_prompt,
    'patching_results': patching_results
}

with open('/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/runs_eval/s2_ioi/c0/scratch/activation_patching.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\n" + "="*80)
print("TOP RECOVERING COMPONENTS")
print("="*80)

sorted_results = sorted(patching_results, key=lambda x: x['recovery'], reverse=True)
for r in sorted_results[:20]:
    print(f"Layer {r['layer']:2d}, {r['component']:5s}: Recovery = {r['recovery']:6.1%}")

print("\nResults saved to scratch/activation_patching.json")
