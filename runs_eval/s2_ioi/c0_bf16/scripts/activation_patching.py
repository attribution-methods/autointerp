"""
Perform activation patching to identify important components for IOI
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

# Load IOI examples
with open('scratch/ioi_verified.json', 'r') as f:
    results = json.load(f)

# Use one example from each template
clean_prompt = results[0]['prompt']  # ABBA
corrupt_prompt = "When Mary and John went to the store, Mary gave a drink to"  # Name flip

io_name = results[0]['io_name']
s_name = results[0]['s_name']

print(f"\nClean prompt:   {clean_prompt}")
print(f"Corrupt prompt: {corrupt_prompt}")
print(f"IO name: {io_name}, S name: {s_name}")

# Get tokens
io_token = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
s_token = tokenizer.encode(" " + s_name, add_special_tokens=False)[0]

def get_logit_diff(prompt):
    """Get the logit difference (IO - S) for a prompt"""
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, -1, :]
    return logits[io_token].item() - logits[s_token].item()

# Get baseline logit diffs
clean_diff = get_logit_diff(clean_prompt)
corrupt_diff = get_logit_diff(corrupt_prompt)

print(f"\nClean logit diff (IO - S): {clean_diff:.3f}")
print(f"Corrupt logit diff (IO - S): {corrupt_diff:.3f}")
print(f"Effect size: {clean_diff - corrupt_diff:.3f}")

# Now perform activation patching across layers
print("\n" + "="*80)
print("ACTIVATION PATCHING BY LAYER")
print("="*80)

n_layers = model.config.n_layer
n_heads = model.config.n_head

# Tokenize both prompts
clean_tokens = tokenizer.encode(clean_prompt, return_tensors="pt").to(handle.device)
corrupt_tokens = tokenizer.encode(corrupt_prompt, return_tensors="pt").to(handle.device)

# Store activations for each layer
def get_activations(prompt):
    """Get residual stream activations at each layer"""
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    activations = []
    
    def hook_fn(module, input, output):
        # output[0] is the hidden states
        activations.append(output[0].detach())
        return output
    
    hooks = []
    for i in range(n_layers):
        hook = model.transformer.h[i].register_forward_hook(hook_fn)
        hooks.append(hook)
    
    with torch.no_grad():
        model(tokens)
    
    for hook in hooks:
        hook.remove()
    
    return activations

print("\nGetting activations for clean and corrupt runs...")
clean_activations = get_activations(clean_prompt)
corrupt_activations = get_activations(corrupt_prompt)

# Patch each layer
print("\nPatching residual stream at each layer...")
patch_results = []

for layer in tqdm(range(n_layers)):
    # Patch by replacing corrupt activation with clean activation at this layer
    def patch_hook(module, input, output):
        output_list = list(output) if isinstance(output, tuple) else [output]
        output_list[0][:, :, :] = clean_activations[layer]
        return tuple(output_list) if isinstance(output, tuple) else output_list[0]
    
    hook = model.transformer.h[layer].register_forward_hook(patch_hook)
    
    patched_diff = get_logit_diff(corrupt_prompt)
    hook.remove()
    
    # Calculate recovery: how much of the effect was recovered?
    recovery = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)
    
    patch_results.append({
        'layer': layer,
        'patched_diff': patched_diff,
        'recovery': recovery
    })

# Display results
print("\n" + "="*80)
print("LAYER PATCHING RESULTS")
print("="*80)
print(f"{'Layer':<8} {'Patched Diff':<15} {'Recovery %':<15}")
print("-" * 80)
for r in patch_results:
    print(f"{r['layer']:<8} {r['patched_diff']:<15.3f} {r['recovery']*100:<15.1f}")

# Save results
with open('scratch/layer_patching_results.json', 'w') as f:
    json.dump({
        'clean_diff': clean_diff,
        'corrupt_diff': corrupt_diff,
        'patch_results': patch_results
    }, f, indent=2)

print("\nSaved to scratch/layer_patching_results.json")

# Identify important layers (recovery > 0.5)
important_layers = [r for r in patch_results if r['recovery'] > 0.5]
print(f"\nImportant layers (recovery > 50%): {[r['layer'] for r in important_layers]}")
