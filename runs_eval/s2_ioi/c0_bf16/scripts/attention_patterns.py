"""
Analyze attention patterns of key heads
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

# Test both ABBA and BABA
examples = [
    ("When Mary and John went to the store, John gave a drink to", "Mary", "John", "ABBA"),
    ("When John and Mary went to the store, Mary gave a drink to", "John", "Mary", "BABA"),
]

for prompt, io_name, s_name, template in examples:
    print("\n" + "="*80)
    print(f"ANALYZING {template} TEMPLATE")
    print("="*80)
    print(f"Prompt: {prompt}")
    print(f"IO (correct): {io_name}, S (subject): {s_name}\n")
    
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    token_strs = tokenizer.convert_ids_to_tokens(tokens[0])
    
    # Print token positions
    for i, tok in enumerate(token_strs):
        print(f"  {i:2d}: {tok}")
    
    # Find key positions
    # First occurrence of IO and S
    io_first_pos = None
    s_first_pos = None
    io_second_pos = None
    s_second_pos = None
    
    for i, tok in enumerate(token_strs):
        # Match with space prefix
        if tok == 'Ġ' + io_name and io_first_pos is None:
            io_first_pos = i
        elif tok == 'Ġ' + io_name and io_first_pos is not None and io_second_pos is None:
            io_second_pos = i
        
        if tok == 'Ġ' + s_name and s_first_pos is None:
            s_first_pos = i
        elif tok == 'Ġ' + s_name and s_first_pos is not None and s_second_pos is None:
            s_second_pos = i
    
    print(f"\nKey positions:")
    print(f"  IO first: {io_first_pos} ({token_strs[io_first_pos] if io_first_pos else 'N/A'})")
    print(f"  S first: {s_first_pos} ({token_strs[s_first_pos] if s_first_pos else 'N/A'})")
    print(f"  IO second: {io_second_pos} ({token_strs[io_second_pos] if io_second_pos else 'N/A'})")
    print(f"  S second: {s_second_pos} ({token_strs[s_second_pos] if s_second_pos else 'N/A'})")
    
    # Cache attention patterns
    attention_patterns = {}
    
    def make_hook(layer, head):
        def hook(module, input, output):
            # output[1] contains attention weights
            attn_weights = output[1]  # [batch, heads, seq, seq]
            if attn_weights is not None:
                attention_patterns[(layer, head)] = attn_weights[0, head, :, :].detach().cpu()
        return hook
    
    hooks = []
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            h = model.transformer.h[layer].attn.register_forward_hook(make_hook(layer, head))
            hooks.append(h)
    
    with torch.no_grad():
        model(tokens, output_attentions=True)
    
    for h in hooks:
        h.remove()
    
    # Analyze key heads (from DLA results)
    key_heads = [
        (11, 0, "positive"),
        (11, 8, "positive"),
        (11, 5, "positive"),
        (11, 2, "positive"),
        (9, 6, "positive"),
        (8, 3, "negative"),
        (10, 2, "negative"),
    ]
    
    seq_len = len(token_strs)
    final_pos = seq_len - 1
    
    print("\n" + "-"*80)
    print("ATTENTION PATTERNS AT FINAL TOKEN")
    print("-"*80)
    
    for layer, head, contrib_type in key_heads:
        if (layer, head) not in attention_patterns:
            continue
        
        attn = attention_patterns[(layer, head)]
        final_attn = attn[final_pos, :].numpy()  # Attention from final token to all positions
        
        print(f"\nL{layer}H{head} ({contrib_type}):")
        
        # Show attention to key positions
        if io_first_pos is not None:
            print(f"  -> IO first ({token_strs[io_first_pos]}): {final_attn[io_first_pos]:.4f}")
        if s_first_pos is not None:
            print(f"  -> S first ({token_strs[s_first_pos]}): {final_attn[s_first_pos]:.4f}")
        if io_second_pos is not None:
            print(f"  -> IO second ({token_strs[io_second_pos] if io_second_pos else 'N/A'}): {final_attn[io_second_pos] if io_second_pos else 'N/A':.4f}")
        if s_second_pos is not None:
            print(f"  -> S second ({token_strs[s_second_pos]}): {final_attn[s_second_pos]:.4f}")
        
        # Show top attended positions
        top_indices = np.argsort(final_attn)[-5:][::-1]
        print(f"  Top 5 attended positions:")
        for idx in top_indices:
            print(f"    {idx:2d}: {token_strs[idx]:15s} {final_attn[idx]:.4f}")
    
    # Create visualization for a few key heads
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Attention Patterns - {template} Template', fontsize=16)
    
    for idx, (layer, head, contrib_type) in enumerate(key_heads[:8]):
        ax = axes[idx // 4, idx % 4]
        
        if (layer, head) in attention_patterns:
            attn = attention_patterns[(layer, head)].numpy()
            
            im = ax.imshow(attn, cmap='viridis', aspect='auto')
            ax.set_xlabel('Source position')
            ax.set_ylabel('Dest position')
            ax.set_title(f'L{layer}H{head} ({contrib_type})')
            
            # Mark key positions
            if io_first_pos: ax.axvline(io_first_pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
            if s_first_pos: ax.axvline(s_first_pos, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            if io_second_pos: ax.axvline(io_second_pos, color='red', linestyle='-', alpha=0.5, linewidth=1)
            if s_second_pos: ax.axvline(s_second_pos, color='blue', linestyle='-', alpha=0.5, linewidth=1)
            
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f'scratch/attention_patterns_{template}.png', dpi=150)
    print(f"\nSaved visualization to scratch/attention_patterns_{template}.png")

print("\n" + "="*80)
print("Analysis complete!")
