"""
Patch individual attention heads to find the critical heads for IOI.
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

clean_prompt = "When John and Mary went to the store, Mary gave a drink to"
corrupt_prompt = "When Mary and John went to the store, Mary gave a drink to"

name_a = "John"
name_b = "Mary"

name_a_id = tokenizer.encode(" " + name_a, add_special_tokens=False)[0]
name_b_id = tokenizer.encode(" " + name_b, add_special_tokens=False)[0]

def compute_logit_diff(logits):
    return logits[name_a_id].item() - logits[name_b_id].item()

clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(handle.device)
corrupt_tokens = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(handle.device)

# Get clean activations at the head level
clean_head_cache = {}

def save_clean_head_output(layer_idx):
    def hook(module, input, output):
        # output is (hidden_states, attention_weights) or just hidden_states
        # We need to capture the output of each head separately
        # In GPT-2, attention heads are computed together and then split
        # So we need to intervene on the attention mechanism
        pass
    return hook

# Actually, let's use a different approach - patch the value projections
# or the attention output for each head

n_layers = model.config.n_layer
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads

print("\n" + "="*80)
print("BASELINE")
print("="*80)

with torch.no_grad():
    clean_output = model(clean_tokens, output_attentions=True)
    clean_logits = clean_output.logits[0, -1, :]
    clean_attentions = clean_output.attentions
    
    corrupt_output = model(corrupt_tokens, output_attentions=True)
    corrupt_logits = corrupt_output.logits[0, -1, :]
    corrupt_attentions = corrupt_output.attentions

clean_logit_diff = compute_logit_diff(clean_logits)
corrupt_logit_diff = compute_logit_diff(corrupt_logits)

print(f"Clean logit diff: {clean_logit_diff:.3f}")
print(f"Corrupt logit diff: {corrupt_logit_diff:.3f}")
print(f"Difference: {clean_logit_diff - corrupt_logit_diff:.3f}")

# Now let's patch individual heads in the key layers (9 and 11)
print("\n" + "="*80)
print("HEAD-LEVEL PATCHING")
print("="*80)

# We'll need to do this more carefully by intercepting at the attention computation level
# Let's cache the attention outputs per head

# Cache clean activations with per-head resolution
clean_head_outputs = {}

def save_head_outputs(layer_idx):
    def hook(module, input, output):
        # output[0] is hidden_states after attention
        # We need to get the output of individual heads before they're combined
        # This requires us to hook into the attention mechanism more deeply
        
        # In transformers, the attention output is computed as:
        # attention_output = self.c_proj(context_layer)
        # where context_layer comes from combining heads
        
        # Let's capture at a different point
        hidden = output[0]  # [batch, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = hidden.shape
        
        # Reshape to separate heads
        # This is the combined output, but we can still work with it
        clean_head_outputs[f'layer_{layer_idx}'] = hidden.detach().clone()
        
    return hook

# Register hooks for layers we care about
hooks = []
for layer_idx in [9, 10, 11]:
    layer = model.transformer.h[layer_idx]
    hooks.append(layer.attn.register_forward_hook(save_head_outputs(layer_idx)))

with torch.no_grad():
    _ = model(clean_tokens)

for hook in hooks:
    hook.remove()

# Now let's try a simpler approach: use the attention pattern analysis
# and direct logit attribution instead

print("\nAnalyzing attention patterns at final token position...")

# For the key layers, let's see what the attention patterns look like
for layer_idx in [9, 10, 11]:
    print(f"\nLayer {layer_idx}:")
    
    clean_attn = clean_attentions[layer_idx][0, :, -1, :]  # [n_heads, seq_len]
    corrupt_attn = corrupt_attentions[layer_idx][0, :, -1, :]
    
    # Identify positions of names
    clean_token_ids = clean_tokens[0].cpu().tolist()
    corrupt_token_ids = corrupt_tokens[0].cpu().tolist()
    
    # Find positions of names in clean prompt
    john_positions = [i for i, t in enumerate(clean_token_ids) if t == name_a_id]
    mary_positions = [i for i, t in enumerate(clean_token_ids) if t == name_b_id]
    
    print(f"  Clean prompt name positions - John: {john_positions}, Mary: {mary_positions}")
    
    for head_idx in range(n_heads):
        clean_attn_to_john = sum(clean_attn[head_idx, pos].item() for pos in john_positions)
        clean_attn_to_mary = sum(clean_attn[head_idx, pos].item() for pos in mary_positions)
        corrupt_attn_to_john = sum(corrupt_attn[head_idx, pos].item() for pos in john_positions)
        corrupt_attn_to_mary = sum(corrupt_attn[head_idx, pos].item() for pos in mary_positions)
        
        # Significant if there's differential attention
        if abs(clean_attn_to_john - clean_attn_to_mary) > 0.1 or abs(corrupt_attn_to_john - corrupt_attn_to_mary) > 0.1:
            print(f"  Head {head_idx}: Clean(John={clean_attn_to_john:.3f}, Mary={clean_attn_to_mary:.3f}), "
                  f"Corrupt(John={corrupt_attn_to_john:.3f}, Mary={corrupt_attn_to_mary:.3f})")

# Let's also do a more detailed per-head patching for layers 9 and 11
print("\n" + "="*80)
print("DETAILED HEAD PATCHING (Layers 9, 11)")
print("="*80)

# This requires implementing head-level intervention
# For now, let's use a proxy: zero out specific heads

patching_results = []

for target_layer in [9, 11]:
    print(f"\nLayer {target_layer}:")
    
    for head_idx in range(n_heads):
        # Create a hook that zeros out a specific head
        def zero_head_hook(module, input, output, target_head=head_idx):
            hidden = output[0]
            batch_size, seq_len, hidden_size = hidden.shape
            
            # Zero out the contribution of this head
            # This is approximate since we're working with combined output
            start_idx = target_head * head_dim
            end_idx = (target_head + 1) * head_dim
            hidden[:, :, start_idx:end_idx] = 0
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        layer = model.transformer.h[target_layer]
        hook = layer.attn.register_forward_hook(zero_head_hook)
        
        with torch.no_grad():
            ablated_logits = model(clean_tokens).logits[0, -1, :]
        
        hook.remove()
        
        ablated_logit_diff = compute_logit_diff(ablated_logits)
        effect = clean_logit_diff - ablated_logit_diff  # Positive means head was helping
        
        patching_results.append({
            'layer': target_layer,
            'head': head_idx,
            'ablated_logit_diff': ablated_logit_diff,
            'effect': effect
        })
        
        if abs(effect) > 0.2:
            print(f"  Head {head_idx}: Effect = {effect:+.3f} (ablated diff = {ablated_logit_diff:.3f})")

# Save results
output_data = {
    'clean_logit_diff': clean_logit_diff,
    'corrupt_logit_diff': corrupt_logit_diff,
    'patching_results': patching_results
}

with open('/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/runs_eval/s2_ioi/c0/scratch/head_patching.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\n" + "="*80)
print("TOP IMPORTANT HEADS (by ablation effect)")
print("="*80)

sorted_results = sorted(patching_results, key=lambda x: abs(x['effect']), reverse=True)
for r in sorted_results[:10]:
    print(f"Layer {r['layer']}, Head {r['head']}: Effect = {r['effect']:+.3f}")

print("\nResults saved to scratch/head_patching.json")
