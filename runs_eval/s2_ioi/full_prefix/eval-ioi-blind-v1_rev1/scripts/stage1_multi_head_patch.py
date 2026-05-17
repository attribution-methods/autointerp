#!/usr/bin/env python3
"""Test multi-head patching to verify combined recovery."""

import sys
import json
from pathlib import Path
import torch
import numpy as np

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

from autointerp.tools.model import load_model
from autointerp.tools.head_patching import (
    cache_head_z, run_with_head_patches, HeadPatch, logit_diff as compute_logit_diff
)

# Load model first
print("Loading model...")
handle = load_model("gpt2", device="cuda", dtype=torch.float32)

# Load samples
with open("scratch/ioi_samples_dev.json") as f:
    clean_samples = json.load(f)
with open("scratch/ioi_samples_dev_corrupted.json") as f:
    corrupt_samples = json.load(f)

# Load previous results
with open("scratch/stage1_head_patch_sweep.json") as f:
    sweep_results = json.load(f)

top_heads = sweep_results["top_heads"][:10]  # Top 10 heads

# Use subset for testing
n_samples = 50
clean_prompts = [s["prompt"] for s in clean_samples[:n_samples]]
corrupt_prompts = [s["prompt"] for s in corrupt_samples[:n_samples]]

# Token IDs
io_token_ids = [handle.tokenizer.encode(" " + s["io_name"], add_special_tokens=False)[0] 
                for s in clean_samples[:n_samples]]
s_token_ids = [handle.tokenizer.encode(" " + s["s_name"], add_special_tokens=False)[0] 
               for s in clean_samples[:n_samples]]

print("Caching activations...")
clean_cache = cache_head_z(handle, clean_prompts)
corrupt_cache = cache_head_z(handle, corrupt_prompts)

# Baseline: compute clean and corrupt logit_diffs
print("Computing baselines...")

def get_logit_diffs(prompts):
    """Run forward pass and compute logit_diffs."""
    tokens = handle.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).to("cuda")
    
    with torch.no_grad():
        outputs = handle.model(**tokens)
        logits = outputs.logits
    
    # Get logits at final position
    seq_lens = tokens["attention_mask"].sum(dim=1)
    final_logits = torch.stack([logits[i, seq_lens[i]-1] for i in range(len(prompts))])
    
    # Compute logit_diff
    diffs = compute_logit_diff(final_logits, io_token_ids, s_token_ids)
    return diffs.cpu().numpy()

clean_diffs = get_logit_diffs(clean_prompts)
corrupt_diffs = get_logit_diffs(corrupt_prompts)

mean_clean = clean_diffs.mean()
mean_corrupt = corrupt_diffs.mean()
gap = mean_clean - mean_corrupt

print(f"Clean logit_diff:   {mean_clean:.4f}")
print(f"Corrupt logit_diff: {mean_corrupt:.4f}")
print(f"Gap:                {gap:.4f}")

# Test multi-head patching
print("\nTesting multi-head patches...")
results = []

for k in [1, 2, 3, 5, 10]:
    sites = [(h["layer"], h["head"]) for h in top_heads[:k]]
    
    # Create patches: patch from clean cache into corrupt run
    patches = []
    for layer, head in sites:
        # Patch at position -1 (END)
        clean_z = clean_cache[layer][:, -1, head, :]  # [B, d_head]
        patches.append(HeadPatch(layer=layer, head=head, position=-1, z=clean_z))
    
    # Run with patches
    patched_logits = run_with_head_patches(handle, corrupt_prompts, patches)
    
    # Get final logits
    tokens = handle.tokenizer(
        corrupt_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).to("cuda")
    seq_lens = tokens["attention_mask"].sum(dim=1)
    final_patched_logits = torch.stack([patched_logits[i, seq_lens[i]-1] 
                                        for i in range(len(corrupt_prompts))])
    
    # Compute patched logit_diff
    patched_diffs = compute_logit_diff(final_patched_logits, io_token_ids, s_token_ids)
    mean_patched = patched_diffs.cpu().mean().item()
    
    # Recovery = (patched - corrupt) / (clean - corrupt)
    recovery = (mean_patched - mean_corrupt) / gap if gap != 0 else 0
    
    print(f"  Top {k:2d} heads: patched={mean_patched:.4f}, recovery={recovery:.4f}")
    results.append({"k": k, "recovery": float(recovery), "heads": [(int(l), int(h)) for l, h in sites]})

# Save results
output = {
    "clean_mean": float(mean_clean),
    "corrupt_mean": float(mean_corrupt),
    "gap": float(gap),
    "multi_head_recovery": results
}

with open("scratch/stage1_multi_head_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved results to scratch/stage1_multi_head_results.json")

# Check criterion
best_recovery = max(r["recovery"] for r in results)
best_k = [r["k"] for r in results if r["recovery"]==best_recovery][0]
print(f"\nBest recovery (top-{best_k}) = {best_recovery:.4f}")
if best_recovery >= 0.85:
    print("✓ Criterion met: patch_effect_recovery >= 0.85")
else:
    print(f"⚠️  Best recovery {best_recovery:.4f} < 0.85")

handle.cleanup()
