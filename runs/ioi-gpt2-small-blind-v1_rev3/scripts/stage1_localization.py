"""Stage 1: Localization via head-level activation patching.

Two-pass approach:
1. Per-head sweep to rank all 144 (layer, head) pairs
2. Joint recovery of top-K heads to feed the criterion
"""
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import json
from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    cache_head_z, run_with_head_patches, HeadPatch, 
    head_patch_sweep, logit_diff
)

def main():
    # Load model
    print("Loading model gpt2...")
    handle = load_model("gpt2")
    n_layers = handle.model.config.n_layer
    n_heads = handle.model.config.n_head
    print(f"Model: {n_layers} layers, {n_heads} heads/layer = {n_layers * n_heads} total heads")
    
    # Generate IOI pairs (dev split)
    print("\nGenerating 500 IOI dev pairs...")
    pairs = generate_pairs(
        spec=None,
        n_pairs=500,
        seed=42,
        tokenizer=handle.tokenizer,
        split="dev"
    )
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    io_ids = torch.tensor([p.target_token_id for p in pairs])
    s_ids = torch.tensor([p.foil_token_id for p in pairs])
    
    print(f"Generated {len(pairs)} pairs")
    print(f"Example clean:   {clean_prompts[0]!r}")
    print(f"Example corrupt: {corrupt_prompts[0]!r}")
    
    # Compute baseline logit_diffs on clean and corrupt
    print("\n=== Computing baselines ===")
    
    # Clean baseline
    clean_tokens = handle.tokenizer(clean_prompts, return_tensors="pt", padding=True, 
                                     add_special_tokens=False).to(handle.device)
    with torch.no_grad():
        clean_logits = handle.model(**clean_tokens).logits[:, -1, :]
    clean_lD = logit_diff(clean_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
    
    # Corrupt baseline
    corrupt_tokens = handle.tokenizer(corrupt_prompts, return_tensors="pt", padding=True,
                                       add_special_tokens=False).to(handle.device)
    with torch.no_grad():
        corrupt_logits = handle.model(**corrupt_tokens).logits[:, -1, :]
    corrupt_lD = logit_diff(corrupt_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
    
    gap = clean_lD - corrupt_lD
    print(f"Clean logit_diff:   {clean_lD:.4f}")
    print(f"Corrupt logit_diff: {corrupt_lD:.4f}")
    print(f"Gap (clean - corrupt): {gap:.4f}")
    
    # Pass 1: Per-head sweep to rank heads
    print("\n=== Pass 1: Per-head sweep across all 144 heads ===")
    print("Caching clean and corrupt head z activations...")
    
    clean_cache = cache_head_z(handle, clean_prompts)
    corrupt_cache = cache_head_z(handle, corrupt_prompts)
    
    print("Running head_patch_sweep...")
    metric_fn = lambda logits: logit_diff(logits, io_ids.to(handle.device), s_ids.to(handle.device))
    
    sweep_results = head_patch_sweep(
        handle, 
        clean_prompts, 
        corrupt_prompts, 
        metric_fn,
        patch_positions=[-1],  # END position only
        clean_cache=clean_cache
    )
    
    recovery_matrix = sweep_results["recovery"]  # [n_layers, n_heads]
    print(f"Recovery matrix shape: {recovery_matrix.shape}")
    print(f"Recovery range: [{recovery_matrix.min():.4f}, {recovery_matrix.max():.4f}]")
    
    # Identify top-K heads
    K = 3
    flat_recovery = recovery_matrix.flatten()
    top_K_indices = torch.topk(flat_recovery, K).indices
    top_K_heads = [(idx.item() // n_heads, idx.item() % n_heads) for idx in top_K_indices]
    top_K_recovery = [recovery_matrix[L, H].item() for L, H in top_K_heads]
    
    print(f"\n=== Top-{K} heads (by individual recovery) ===")
    for i, ((L, H), rec) in enumerate(zip(top_K_heads, top_K_recovery)):
        print(f"  {i+1}. Layer {L}, Head {H}: recovery = {rec:.4f}")
    
    # Pass 2: Joint recovery by patching all K heads simultaneously
    print(f"\n=== Pass 2: Joint recovery (patching all {K} heads together) ===")
    
    # Build patches for all top-K heads from clean cache
    patches = []
    for L, H in top_K_heads:
        # clean_cache[L] has shape [batch, seq, n_heads, d_head]
        # Extract head H's z values from clean cache
        source_z = clean_cache[L][:, :, H, :]  # [batch, seq, d_head]
        patches.append(HeadPatch(layer=L, head=H, source=source_z, positions=[-1]))
    
    print(f"Patching {len(patches)} heads into corrupt run...")
    patched_logits = run_with_head_patches(
        handle, 
        corrupt_prompts, 
        patches, 
        return_logits_at=-1
    )
    
    joint_lD = logit_diff(patched_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
    joint_recovery = (joint_lD - corrupt_lD) / gap
    
    print(f"Patched logit_diff (joint): {joint_lD:.4f}")
    print(f"Joint recovery: {joint_recovery:.4f}")
    print(f"Gap recovered: {(joint_lD - corrupt_lD):.4f} / {gap:.4f}")
    
    # Save results
    results = {
        "clean_logit_diff": float(clean_lD),
        "corrupt_logit_diff": float(corrupt_lD),
        "gap": float(gap),
        "recovery_matrix": recovery_matrix.tolist(),
        "top_K": K,
        "top_K_heads": [[int(L), int(H)] for L, H in top_K_heads],
        "top_K_individual_recovery": [float(r) for r in top_K_recovery],
        "joint_logit_diff": float(joint_lD),
        "joint_recovery": float(joint_recovery),
        "criterion_inputs": {
            "baseline_metric": float(clean_lD),
            "corrupted_metric": float(corrupt_lD),
            "patched_metric": float(joint_lD)
        }
    }
    
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Top-{K} heads identified: {top_K_heads}")
    print(f"Joint recovery: {joint_recovery:.4f}")
    print(f"Criterion threshold: 0.85")
    print(f"Pass: {joint_recovery >= 0.85}")
    print("\nResults saved to scratch/stage1_results.json")
    
    return joint_recovery

if __name__ == "__main__":
    main()
