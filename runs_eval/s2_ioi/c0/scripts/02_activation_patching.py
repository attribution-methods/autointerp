"""
Activation patching to identify critical layers and components for IOI.
We'll patch from a "corrupted" version to a clean version to see which 
components are necessary.
"""
import sys
sys.path.insert(0, '/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp')

import torch
import numpy as np
from autointerp.tools.model import load_model
from autointerp.tools.patching import sweep_patch_sites
import json

# Load GPT-2-small
print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

# Use a clean and corrupted version of the same template
# Clean: The model predicts the correct indirect object (John)
# Corrupted: We'll swap the names to see if the model now predicts Mary
clean_prompt = "When John and Mary went to the store, Mary gave a drink to"
corrupt_prompt = "When Mary and John went to the store, Mary gave a drink to"

# The target names
name_a = "John"  # Indirect object - should get higher logit
name_b = "Mary"  # Subject - should get lower logit

# Get token IDs
name_a_id = tokenizer.encode(" " + name_a, add_special_tokens=False)[0]
name_b_id = tokenizer.encode(" " + name_b, add_special_tokens=False)[0]

def compute_logit_diff(logits):
    """Compute logit difference between name A and name B."""
    return logits[name_a_id].item() - logits[name_b_id].item()

# Baseline: compute logit diff for clean and corrupt
print("\n" + "="*80)
print("BASELINE LOGIT DIFFERENCES")
print("="*80)

clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(handle.device)
corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(handle.device)

with torch.no_grad():
    clean_logits = model(clean_tokens).logits[0, -1, :]
    corrupt_logits = model(corrupt_tokens).logits[0, -1, :]

clean_logit_diff = compute_logit_diff(clean_logits)
corrupt_logit_diff = compute_logit_diff(corrupt_logits)

print(f"Clean prompt: {clean_prompt}")
print(f"Clean logit diff (A - B): {clean_logit_diff:.3f}")
print(f"\nCorrupt prompt: {corrupt_prompt}")
print(f"Corrupt logit diff (A - B): {corrupt_logit_diff:.3f}")
print(f"\nDifference: {clean_logit_diff - corrupt_logit_diff:.3f}")

# Now let's patch different components
print("\n" + "="*80)
print("ACTIVATION PATCHING SWEEP")
print("="*80)

# We'll patch residual stream, attention output, and MLP output at each layer
n_layers = model.config.n_layer
n_heads = model.config.n_head

# Create sites to patch
sites = []

# Residual stream at each layer
for layer in range(n_layers):
    sites.append({
        'layer': layer,
        'component': 'resid',
        'token_index': -1  # Last token
    })

# Attention output at each layer
for layer in range(n_layers):
    sites.append({
        'layer': layer,
        'component': 'attn',
        'token_index': -1
    })

# MLP output at each layer
for layer in range(n_layers):
    sites.append({
        'layer': layer,
        'component': 'mlp',
        'token_index': -1
    })

print(f"Testing {len(sites)} patch sites...")
print("This may take a few minutes...")

# Run the sweep
try:
    results = sweep_patch_sites(
        handle=handle,
        clean=clean_prompt,
        corrupt=corrupt_prompt,
        sites=sites
    )
    
    print("\n" + "="*80)
    print("PATCHING RESULTS")
    print("="*80)
    
    # Process results
    patching_results = []
    
    for site, result in zip(sites, results):
        layer = site['layer']
        component = site['component']
        
        # Get the logits after patching
        patched_logits = result['logits']
        patched_logit_diff = compute_logit_diff(patched_logits)
        
        # Recovery: how much did patching recover the clean behavior?
        recovery = (patched_logit_diff - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)
        
        patching_results.append({
            'layer': layer,
            'component': component,
            'patched_logit_diff': patched_logit_diff,
            'recovery': recovery
        })
        
        if recovery > 0.1:  # Only print significant recoveries
            print(f"Layer {layer:2d}, {component:5s}: Recovery = {recovery:6.1%}, Logit diff = {patched_logit_diff:.3f}")
    
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
    
    print("\nResults saved to scratch/activation_patching.json")
    
    # Summary: top recovering components
    print("\n" + "="*80)
    print("TOP RECOVERING COMPONENTS")
    print("="*80)
    
    sorted_results = sorted(patching_results, key=lambda x: x['recovery'], reverse=True)
    for r in sorted_results[:15]:
        print(f"Layer {r['layer']:2d}, {r['component']:5s}: Recovery = {r['recovery']:6.1%}")

except Exception as e:
    print(f"Error during patching sweep: {e}")
    import traceback
    traceback.print_exc()
