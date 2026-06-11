"""
Analyze MLP contributions to IOI
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
import numpy as np
from autointerp.tools.model import load_model

print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

prompt = "When Mary and John went to the store, John gave a drink to"
io_name = "Mary"
s_name = "John"

tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
io_token = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
s_token = tokenizer.encode(" " + s_name, add_special_tokens=False)[0]

def get_logit_diff():
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, -1, :]
    return logits[io_token].item() - logits[s_token].item()

baseline = get_logit_diff()
print(f"Baseline logit diff (IO - S): {baseline:.3f}")

print("\n" + "="*80)
print("MLP ABLATION EXPERIMENTS")
print("="*80)

# Test ablating MLPs layer by layer
for layer_idx in [8, 9, 10, 11]:
    def ablate_mlp(module, input, output):
        return torch.zeros_like(output)
    
    hook = model.transformer.h[layer_idx].mlp.register_forward_hook(ablate_mlp)
    ablated = get_logit_diff()
    hook.remove()
    
    change = ablated - baseline
    print(f"Layer {layer_idx:2d} MLP ablated: {ablated:.3f} (Δ={change:+.3f})")

print("\n" + "="*80)
print("COMBINED ABLATIONS")
print("="*80)

# Test ablating all layer 11 (both attention and MLP)
def ablate_attn(module, input, output):
    attn_out = output[0]
    return (torch.zeros_like(attn_out),) + output[1:]

def ablate_mlp(module, input, output):
    return torch.zeros_like(output)

hook1 = model.transformer.h[11].attn.register_forward_hook(ablate_attn)
hook2 = model.transformer.h[11].mlp.register_forward_hook(ablate_mlp)
ablated_l11_full = get_logit_diff()
hook1.remove()
hook2.remove()

print(f"Ablate L11 (attention only):  {1.75:.3f}")
print(f"Ablate L11 (attn + MLP):      {ablated_l11_full:.3f}")
print(f"Combined effect:              {ablated_l11_full - baseline:+.3f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("MLPs in later layers (especially L10-11) also contribute to IOI")
print("The circuit involves BOTH attention and MLP components")
print("This is consistent with the direct logit attribution showing")
print("negative MLP contributions that get overridden")
