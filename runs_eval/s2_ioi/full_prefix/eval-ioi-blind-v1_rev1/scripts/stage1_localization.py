#!/usr/bin/env python3
"""Stage 1: Localization via logit lens and head-level activation patching."""

import sys
import json
import random
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

from autointerp.tools.model import load_model
from autointerp.tools.lenses import logit_lens
from autointerp.tools.head_patching import (
    cache_head_z, mean_head_z, head_patch_sweep, logit_diff as compute_logit_diff
)

# Load IOI samples from Stage 0
with open("scratch/ioi_samples_dev.json") as f:
    clean_samples = json.load(f)

print(f"Loaded {len(clean_samples)} clean samples")

# Generate corrupted samples (ABC pattern)
# Clean: "When A and B went to park, A/B gave a book to" -> IO is the non-repeated name
# Corrupt: "When A and C went to park, A/B gave a book to" -> replace one name with novel C

def generate_corrupted_samples(clean_samples, seed=42):
    """Generate ABC corrupted samples."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Names pool for generating C (should not overlap with A or B)
    all_names = ["John", "Mary", "Tom", "Sarah", "James", "Kate", "Robert", "Lisa",
                 "Michael", "Emma", "David", "Sophia", "Chris", "Anna", "Mark", "Grace",
                 "Peter", "Linda", "Paul", "Nancy", "George", "Helen"]
    
    corrupted = []
    for sample in clean_samples:
        # Parse the clean prompt to extract A, B
        prompt = sample["prompt"]
        io_name = sample["io_name"]
        s_name = sample["s_name"]
        
        # Find a C that's different from both A and B
        used_names = {io_name, s_name}
        available = [n for n in all_names if n not in used_names]
        c_name = random.choice(available)
        
        # Replace S with C in the prompt (S is the repeated/subject name)
        # In ABBA: "When A and B..., A gave..." -> S=B appears once, replace with C
        # In BABA: "When A and B..., B gave..." -> S=B appears twice, replace first occurrence
        
        if sample["template"] == "ABBA":
            # "When A and B went to park, A gave..." -> replace B with C
            corrupt_prompt = prompt.replace(f" {s_name} ", f" {c_name} ", 1)
        else:  # BABA
            # "When A and B went to park, B gave..." -> replace first B with C
            corrupt_prompt = prompt.replace(f"When {io_name} and {s_name}", 
                                          f"When {io_name} and {c_name}", 1)
        
        corrupted.append({
            "prompt": corrupt_prompt,
            "io_name": io_name,
            "s_name": s_name,
            "c_name": c_name,
            "template": sample["template"],
            "clean_prompt": prompt
        })
    
    return corrupted

print("Generating corrupted (ABC) samples...")
corrupted_samples = generate_corrupted_samples(clean_samples, seed=42)

# Save corrupted samples
with open("scratch/ioi_samples_dev_corrupted.json", "w") as f:
    json.dump(corrupted_samples, f, indent=2)

print(f"Generated {len(corrupted_samples)} corrupted samples")
print(f"Example clean:     {clean_samples[0]['prompt']}")
print(f"Example corrupted: {corrupted_samples[0]['prompt']}")

# Load model
print("\nLoading GPT-2...")
handle = load_model("gpt2", device="cuda", dtype=torch.float32)
print(f"Model: {handle.model_id}, layers={handle.n_layers}, heads={handle.n_heads}")

# Part 1: Logit Lens Analysis
print("\n" + "="*60)
print("PART 1: LOGIT LENS ANALYSIS")
print("="*60)

# Run logit lens on a subset of samples at END position
n_lens_samples = 20
lens_samples = clean_samples[:n_lens_samples]

logit_lens_results = []
for i, sample in enumerate(tqdm(lens_samples, desc="Logit lens")):
    prompt = sample["prompt"]
    io_name = sample["io_name"]
    s_name = sample["s_name"]
    
    # Get logit lens predictions across all layers
    # The logit_lens function should return predictions at each layer
    # We want to see at which layer the IO name starts appearing in top predictions
    
    # For now, save the prompt and names for analysis
    logit_lens_results.append({
        "prompt": prompt,
        "io_name": io_name,
        "s_name": s_name
    })

# TODO: Actually implement logit lens - will do after head patching
print(f"Logit lens: analyzed {len(logit_lens_results)} samples")

# Part 2: Head-Level Activation Patching
print("\n" + "="*60)
print("PART 2: HEAD-LEVEL ACTIVATION PATCHING")
print("="*60)

# Use subset for head patching (full sweep is expensive)
n_patch_samples = 50
clean_prompts = [s["prompt"] for s in clean_samples[:n_patch_samples]]
corrupt_prompts = [s["prompt"] for s in corrupted_samples[:n_patch_samples]]

# Get IO and S token IDs for each sample
io_token_ids = []
s_token_ids = []
for sample in clean_samples[:n_patch_samples]:
    io_tok = handle.tokenizer.encode(" " + sample["io_name"], add_special_tokens=False)[0]
    s_tok = handle.tokenizer.encode(" " + sample["s_name"], add_special_tokens=False)[0]
    io_token_ids.append(io_tok)
    s_token_ids.append(s_tok)

print(f"Patching {n_patch_samples} samples...")
print(f"Clean prompts: {len(clean_prompts)}")
print(f"Corrupt prompts: {len(corrupt_prompts)}")

# Cache head z activations
print("Caching clean activations...")
clean_cache = cache_head_z(handle, clean_prompts)
print(f"Clean cache: {len(clean_cache)} layers")

print("Caching corrupt activations...")
corrupt_cache = cache_head_z(handle, corrupt_prompts)
print(f"Corrupt cache: {len(corrupt_cache)} layers")

# Define metric: logit_diff = logit(IO) - logit(S)
def metric(logits):
    """Compute logit_diff for a batch of logits."""
    return compute_logit_diff(logits, io_token_ids, s_token_ids)

# Run head patch sweep
print("Running head patch sweep...")
print(f"Sweeping {handle.n_layers} layers x {handle.n_heads} heads = {handle.n_layers * handle.n_heads} sites")

sweep_result = head_patch_sweep(
    handle,
    clean_prompts,
    corrupt_prompts,
    metric,
    patch_positions=[-1],  # Patch only at END position
    clean_cache=clean_cache
)

recovery = sweep_result["recovery"]  # [L, H] tensor
print(f"\nRecovery shape: {recovery.shape}")

# Convert to numpy and analyze
recovery_np = recovery.cpu().numpy()

# Find top heads
flat_recovery = recovery_np.flatten()
top_k = 20
top_indices = np.argsort(flat_recovery)[-top_k:][::-1]
top_heads = [(idx // handle.n_heads, idx % handle.n_heads, flat_recovery[idx]) 
             for idx in top_indices]

print(f"\nTop {top_k} heads by recovery:")
for layer, head, rec in top_heads:
    print(f"  L{layer}H{head}: {rec:.4f}")

# Save results
results = {
    "recovery_matrix": recovery_np.tolist(),
    "top_heads": [{"layer": int(l), "head": int(h), "recovery": float(r)} 
                  for l, h, r in top_heads],
    "n_samples": n_patch_samples,
    "n_layers": handle.n_layers,
    "n_heads": handle.n_heads
}

with open("scratch/stage1_head_patch_sweep.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved sweep results to scratch/stage1_head_patch_sweep.json")

# Summary statistics
mean_recovery = float(np.mean(recovery_np))
max_recovery = float(np.max(recovery_np))
print(f"\nRecovery statistics:")
print(f"  Mean: {mean_recovery:.4f}")
print(f"  Max:  {max_recovery:.4f}")
print(f"  Top head: L{top_heads[0][0]}H{top_heads[0][1]} = {top_heads[0][2]:.4f}")

handle.cleanup()
print("\nDone!")
