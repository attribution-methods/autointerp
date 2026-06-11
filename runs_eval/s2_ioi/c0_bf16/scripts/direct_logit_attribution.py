"""
Use direct logit attribution to identify which heads contribute to IOI
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
import json
import numpy as np
from autointerp.tools.model import load_model
from autointerp.tools.lenses import direct_logit_attribution
import matplotlib.pyplot as plt

# Load model
print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

# Use ABBA example
prompt = "When Mary and John went to the store, John gave a drink to"
io_name = "Mary"
s_name = "John"

print(f"Prompt: {prompt}")
print(f"IO (correct): {io_name}, S (incorrect): {s_name}")

# Tokenize
tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
token_strs = tokenizer.convert_ids_to_tokens(tokens[0])
print(f"\nTokens: {token_strs}")

# Get token IDs for IO and S
io_token_id = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
s_token_id = tokenizer.encode(" " + s_name, add_special_tokens=False)[0]

# Run model and collect activations
n_layers = model.config.n_layer
n_heads = model.config.n_head
d_model = model.config.n_embd

# Get embeddings and layer outputs
activations = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        else:
            activations[name] = output.detach()
    return hook

# Hook embeddings
model.transformer.wte.register_forward_hook(make_hook('embed'))

# Hook each layer's output (after layernorm)
for i in range(n_layers):
    model.transformer.h[i].register_forward_hook(make_hook(f'layer_{i}'))

# Run forward pass
with torch.no_grad():
    outputs = model(tokens)
    final_logits = outputs.logits[0, -1, :]

# Get logit difference
io_logit = final_logits[io_token_id].item()
s_logit = final_logits[s_token_id].item()
logit_diff = io_logit - s_logit

print(f"\nIO logit: {io_logit:.3f}")
print(f"S logit: {s_logit:.3f}")
print(f"Logit diff (IO - S): {logit_diff:.3f}")

# Now decompose contributions by layer
# Each layer transforms: x_out = x_in + attn_out + mlp_out
# The contribution to final logits is: (attn_out @ W_U + mlp_out @ W_U)

# Get the unembedding matrix
W_U = model.lm_head.weight.T  # shape: [d_model, vocab_size]

# For each layer, extract attention and MLP contributions
layer_contributions = []

for layer_idx in range(n_layers):
    layer = model.transformer.h[layer_idx]
    
    # Get input to this layer
    if layer_idx == 0:
        layer_input = activations['embed'][0, -1, :]  # final token
    else:
        layer_input = activations[f'layer_{layer_idx-1}'][0, -1, :]
    
    # Run through attention and MLP separately
    with torch.no_grad():
        # Attention
        ln1_out = layer.ln_1(layer_input.unsqueeze(0).unsqueeze(0))
        attn_out, _ = layer.attn(ln1_out)
        attn_out = attn_out[0, 0, :]  # [d_model]
        
        # MLP
        attn_resid = layer_input + attn_out
        ln2_out = layer.ln_2(attn_resid.unsqueeze(0).unsqueeze(0))
        mlp_out = layer.mlp(ln2_out)[0, 0, :]  # [d_model]
    
    # Project to logits
    attn_logits = attn_out @ W_U
    mlp_logits = mlp_out @ W_U
    
    # Get contributions to IO and S tokens
    attn_io = attn_logits[io_token_id].item()
    attn_s = attn_logits[s_token_id].item()
    attn_diff = attn_io - attn_s
    
    mlp_io = mlp_logits[io_token_id].item()
    mlp_s = mlp_logits[s_token_id].item()
    mlp_diff = mlp_io - mlp_s
    
    layer_contributions.append({
        'layer': layer_idx,
        'attn_diff': attn_diff,
        'mlp_diff': mlp_diff,
        'total_diff': attn_diff + mlp_diff
    })

# Display results
print("\n" + "="*80)
print("DIRECT LOGIT ATTRIBUTION BY LAYER")
print("="*80)
print(f"{'Layer':<8} {'Attn Contrib':<15} {'MLP Contrib':<15} {'Total':<15}")
print("-" * 80)
for c in layer_contributions:
    print(f"{c['layer']:<8} {c['attn_diff']:<15.3f} {c['mlp_diff']:<15.3f} {c['total_diff']:<15.3f}")

print(f"\nSum of contributions: {sum(c['total_diff'] for c in layer_contributions):.3f}")
print(f"Actual logit diff: {logit_diff:.3f}")

# Save results
with open('scratch/layer_contributions.json', 'w') as f:
    json.dump({
        'logit_diff': logit_diff,
        'layer_contributions': layer_contributions
    }, f, indent=2)

print("Saved to scratch/layer_contributions.json")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

layers = [c['layer'] for c in layer_contributions]
attn_diffs = [c['attn_diff'] for c in layer_contributions]
mlp_diffs = [c['mlp_diff'] for c in layer_contributions]
total_diffs = [c['total_diff'] for c in layer_contributions]

axes[0].bar(layers, attn_diffs, color='blue', alpha=0.7)
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Contribution to IO-S logit diff')
axes[0].set_title('Attention Contributions')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

axes[1].bar(layers, mlp_diffs, color='green', alpha=0.7)
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Contribution to IO-S logit diff')
axes[1].set_title('MLP Contributions')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

axes[2].bar(layers, total_diffs, color='purple', alpha=0.7)
axes[2].set_xlabel('Layer')
axes[2].set_ylabel('Contribution to IO-S logit diff')
axes[2].set_title('Total Contributions')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('scratch/layer_contributions.png', dpi=150)
print("Saved plot to scratch/layer_contributions.png")
