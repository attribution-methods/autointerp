"""
Analyze the S-inhibition mechanism: heads that attend to S2 and
suppress it from being predicted.
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
name_a = "John"  # Indirect object (IO)
name_b = "Mary"  # Subject (S) - appears twice, S1 and S2

name_a_id = tokenizer.encode(" " + name_a, add_special_tokens=False)[0]
name_b_id = tokenizer.encode(" " + name_b, add_special_tokens=False)[0]

clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(handle.device)
clean_token_list = clean_tokens[0].cpu().tolist()

# Find positions
name_b_positions = [i for i, t in enumerate(clean_token_list) if t == name_b_id]
s1_pos = name_b_positions[0]  # First occurrence (position 3)
s2_pos = name_b_positions[1]  # Second occurrence (position 9)
io_pos = [i for i, t in enumerate(clean_token_list) if t == name_a_id][0]

print("\n" + "="*80)
print("S-INHIBITION ANALYSIS")
print("="*80)
print(f"Prompt: {clean_prompt}")
print(f"IO position ('{name_a}'): {io_pos}")
print(f"S1 position ('{name_b}'): {s1_pos}")
print(f"S2 position ('{name_b}'): {s2_pos}")

# Get model outputs with attention
with torch.no_grad():
    outputs = model(clean_tokens, output_attentions=True, output_hidden_states=True)
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    final_logits = outputs.logits[0, -1, :]

# Identify heads that strongly attend to S2 from the final position
print("\n" + "="*80)
print("HEADS ATTENDING TO S2 FROM FINAL POSITION")
print("="*80)

s2_attending_heads = []
for layer in range(model.config.n_layer):
    for head in range(model.config.n_head):
        attn_to_s2 = attentions[layer][0, head, -1, s2_pos].item()
        if attn_to_s2 > 0.2:
            s2_attending_heads.append((layer, head, attn_to_s2))
            print(f"L{layer:2d}H{head:2d}: {attn_to_s2:.3f}")

# Now let's check if these heads have negative contribution to Mary's logit
print("\n" + "="*80)
print("CHECKING NEGATIVE CONTRIBUTIONS FOR S2-ATTENDING HEADS")
print("="*80)

# We need to get per-head contributions
# This requires decomposing the attention output by head

W_U = model.lm_head.weight  # [vocab_size, hidden_size]

for layer, head, attn_weight in s2_attending_heads[:10]:
    # Approximate per-head contribution
    # Get the attention output for this layer
    layer_module = model.transformer.h[layer]
    
    # We need to recompute the head's contribution
    # For now, let's ablate the head and see the effect
    
    def zero_head_hook(module, input, output, target_head=head):
        hidden = output[0]
        batch_size, seq_len, hidden_size = hidden.shape
        head_dim = hidden_size // model.config.n_head
        
        # Zero out this head's contribution
        start_idx = target_head * head_dim
        end_idx = (target_head + 1) * head_dim
        hidden[:, :, start_idx:end_idx] = 0
        
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    
    hook = layer_module.attn.register_forward_hook(zero_head_hook)
    
    with torch.no_grad():
        ablated_logits = model(clean_tokens).logits[0, -1, :]
    
    hook.remove()
    
    # Compute change in Mary's logit
    original_logit_b = final_logits[name_b_id].item()
    ablated_logit_b = ablated_logits[name_b_id].item()
    effect_on_b = ablated_logit_b - original_logit_b
    
    # Also check John's logit
    original_logit_a = final_logits[name_a_id].item()
    ablated_logit_a = ablated_logits[name_a_id].item()
    effect_on_a = ablated_logit_a - original_logit_a
    
    print(f"\nL{layer:2d}H{head:2d} (attn to S2: {attn_weight:.3f}):")
    print(f"  Effect on '{name_b}' logit: {effect_on_b:+.3f} (was {original_logit_b:.2f}, now {ablated_logit_b:.2f})")
    print(f"  Effect on '{name_a}' logit: {effect_on_a:+.3f} (was {original_logit_a:.2f}, now {ablated_logit_a:.2f})")
    print(f"  Effect on logit diff (A-B): {(effect_on_a - effect_on_b):+.3f}")
    
    if effect_on_b > 0:
        print(f"  → INHIBITION: Removing this head increases Mary's logit (head was suppressing it)")

print("\n" + "="*80)
print("DUPLICATE TOKEN HEADS (S2 attending to S1)")
print("="*80)

# Look for heads where S2 attends to S1
duplicate_token_heads = []
for layer in range(model.config.n_layer):
    for head in range(model.config.n_head):
        attn_s2_to_s1 = attentions[layer][0, head, s2_pos, s1_pos].item()
        if attn_s2_to_s1 > 0.3:
            duplicate_token_heads.append((layer, head, attn_s2_to_s1))
            print(f"L{layer:2d}H{head:2d}: S2→S1 attention = {attn_s2_to_s1:.3f}")

print("\n" + "="*80)
print("NAME MOVER HEADS (Final position attending to IO)")
print("="*80)

# These should be the heads that copy the IO token to the output
name_mover_heads = []
for layer in range(model.config.n_layer):
    for head in range(model.config.n_head):
        attn_to_io = attentions[layer][0, head, -1, io_pos].item()
        if attn_to_io > 0.3:
            name_mover_heads.append((layer, head, attn_to_io))
            print(f"L{layer:2d}H{head:2d}: Final→IO attention = {attn_to_io:.3f}")

# Test the effect of these heads
print("\n" + "="*80)
print("EFFECT OF NAME MOVER HEADS ON OUTPUT")
print("="*80)

for layer, head, attn_weight in name_mover_heads[:5]:
    def zero_head_hook(module, input, output, target_head=head):
        hidden = output[0]
        batch_size, seq_len, hidden_size = hidden.shape
        head_dim = hidden_size // model.config.n_head
        
        start_idx = target_head * head_dim
        end_idx = (target_head + 1) * head_dim
        hidden[:, :, start_idx:end_idx] = 0
        
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    
    layer_module = model.transformer.h[layer]
    hook = layer_module.attn.register_forward_hook(zero_head_hook)
    
    with torch.no_grad():
        ablated_logits = model(clean_tokens).logits[0, -1, :]
    
    hook.remove()
    
    original_logit_diff = final_logits[name_a_id].item() - final_logits[name_b_id].item()
    ablated_logit_diff = ablated_logits[name_a_id].item() - ablated_logits[name_b_id].item()
    effect = original_logit_diff - ablated_logit_diff
    
    print(f"\nL{layer:2d}H{head:2d} (attn to IO: {attn_weight:.3f}):")
    print(f"  Original logit diff (A-B): {original_logit_diff:.3f}")
    print(f"  Ablated logit diff (A-B): {ablated_logit_diff:.3f}")
    print(f"  Effect: {effect:+.3f}")
    if effect > 0:
        print(f"  → NAME MOVER: This head promotes IO over S")

print("\nAnalysis complete!")
