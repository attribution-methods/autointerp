"""
Direct logit attribution for individual attention heads
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
io_name = "Mary"
s_name = "John"

tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
io_token_id = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
s_token_id = tokenizer.encode(" " + s_name, add_special_tokens=False)[0]

n_layers = model.config.n_layer
n_heads = model.config.n_head
d_model = model.config.n_embd
d_head = d_model // n_heads

# Get W_U (unembedding)
W_U = model.lm_head.weight.T  # [d_model, vocab]

# Cache all intermediate activations
cache = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache[name] = output[0].detach()
        else:
            cache[name] = output.detach()
    return hook

hooks = []

# Hook layer inputs
for i in range(n_layers):
    h = model.transformer.h[i].register_forward_hook(make_hook(f'layer_{i}_in'))
    hooks.append(h)

# Run forward pass
with torch.no_grad():
    outputs = model(tokens)
    final_logits = outputs.logits[0, -1, :]

for h in hooks:
    h.remove()

logit_diff = final_logits[io_token_id].item() - final_logits[s_token_id].item()
print(f"Logit diff (IO - S): {logit_diff:.3f}")

# Now for each layer, decompose attention into heads
head_contributions = []

for layer_idx in range(n_layers):
    layer = model.transformer.h[layer_idx]
    
    # Get the input to this layer
    layer_input = cache[f'layer_{layer_idx}_in'][0, -1, :]  # [d_model]
    
    # Pass through layer norm 1
    with torch.no_grad():
        ln1_out = layer.ln_1(layer_input.unsqueeze(0).unsqueeze(0))[0, 0, :]  # [d_model]
    
    # Get attention weights and values
    # We need to manually decompose the attention into heads
    with torch.no_grad():
        # Project to Q, K, V
        qkv = layer.attn.c_attn(ln1_out.unsqueeze(0).unsqueeze(0))  # [1, 1, 3*d_model]
        q, k, v = qkv.split(d_model, dim=2)
        
        # Reshape to separate heads: [batch, seq, heads, d_head]
        q = q.view(1, 1, n_heads, d_head).transpose(1, 2)  # [1, heads, 1, d_head]
        k = k.view(1, 1, n_heads, d_head).transpose(1, 2)
        v = v.view(1, 1, n_heads, d_head).transpose(1, 2)
        
        # For each head, compute its contribution
        for head in range(n_heads):
            # Get this head's output: attn_weights @ v
            # Since we only care about the final position, we can simplify
            # But actually we need full context for attention
            # Let me re-run with full sequence
            pass

# Actually, let me use a cleaner approach - cache attention patterns
print("\nCaching attention outputs by head...")

head_outputs = {}  # (layer, head) -> output vector

for layer_idx in range(n_layers):
    layer = model.transformer.h[layer_idx]
    
    # Get input to this layer
    if layer_idx == 0:
        # Get embeddings
        with torch.no_grad():
            inputs_embeds = model.transformer.wte(tokens)
            position_ids = torch.arange(0, tokens.shape[1], dtype=torch.long, device=tokens.device)
            position_embeds = model.transformer.wpe(position_ids)
            layer_input = inputs_embeds + position_embeds
    else:
        layer_input = cache[f'layer_{layer_idx}_in']
    
    # Apply layer norm 1
    with torch.no_grad():
        attn_input = layer.ln_1(layer_input)  # [1, seq, d_model]
        
        # Run attention manually to extract per-head outputs
        batch_size, seq_len, _ = attn_input.shape
        
        # Project to Q, K, V
        qkv = layer.attn.c_attn(attn_input)
        query, key, value = qkv.split(d_model, dim=2)
        
        # Reshape to heads
        query = query.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
        
        # Attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.sqrt(torch.tensor(d_head, dtype=torch.float32))
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)  # [batch, heads, seq, d_head]
        
        # For each head, get output at final position
        final_pos = seq_len - 1
        for head in range(n_heads):
            head_out = attn_output[0, head, final_pos, :]  # [d_head]
            
            # Project through W_O
            # The c_proj combines all heads, so we need to extract the right slice
            W_O = layer.attn.c_proj.weight.T  # [d_model, d_model]
            W_O_head = W_O[head * d_head:(head + 1) * d_head, :]  # [d_head, d_model]
            
            # Project head output to residual stream
            head_contribution = head_out @ W_O_head  # [d_model]
            
            # Project to logits
            head_logits = head_contribution @ W_U  # [vocab]
            
            # Get contribution to IO - S
            contrib = head_logits[io_token_id].item() - head_logits[s_token_id].item()
            
            head_outputs[(layer_idx, head)] = contrib

# Display top contributing heads
contributions = [(k, v) for k, v in head_outputs.items()]
contributions.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n" + "="*80)
print("TOP 30 ATTENTION HEADS BY CONTRIBUTION TO IO-S LOGIT DIFF")
print("="*80)
print(f"{'Layer':<8} {'Head':<8} {'Contribution':<15}")
print("-" * 80)
for (layer, head), contrib in contributions[:30]:
    print(f"{layer:<8} {head:<8} {contrib:<15.3f}")

# Save
with open('scratch/head_dla.json', 'w') as f:
    json.dump({
        'logit_diff': logit_diff,
        'head_contributions': {f'L{k[0]}H{k[1]}': v for k, v in head_outputs.items()}
    }, f, indent=2)

# Create heatmap
contrib_matrix = np.zeros((n_layers, n_heads))
for (layer, head), contrib in head_outputs.items():
    contrib_matrix[layer, head] = contrib

plt.figure(figsize=(12, 8))
plt.imshow(contrib_matrix, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
plt.colorbar(label='Contribution to IO-S logit diff')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.title('Attention Head Contributions to IOI (Direct Logit Attribution)')
plt.tight_layout()
plt.savefig('scratch/head_dla_heatmap.png', dpi=150)
print("\nSaved heatmap to scratch/head_dla_heatmap.png")
