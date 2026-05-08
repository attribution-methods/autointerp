#!/usr/bin/env python3
"""
Stage 2: Exhaustive head_patch_sweep localization.
Run sweep across all 144 (layer, head) pairs on dev split.
"""

import sys
sys.path.insert(0, "src")

import torch
import json
import numpy as np
from pathlib import Path

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    cache_head_z, head_patch_sweep, logit_diff
)

def main():
    print("=" * 80)
    print("STAGE 2: EXHAUSTIVE HEAD_PATCH_SWEEP")
    print("=" * 80)
    
    # Load spec
    spec_path = Path("runs/ioi-discovery-v1_rev3/spec.json")
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load model
    print("\n[1/5] Loading GPT-2-small...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded: {handle.model_id}, n_layers={handle.n_layers}, n_heads={handle.n_heads}")
    print(f"Total (layer, head) pairs: {handle.n_layers * handle.n_heads}")
    
    # Load IOI dataset
    print("\n[2/5] Loading IOI dev dataset (n=500, seed=42)...")
    pairs = generate_pairs(
        spec,
        n_pairs=500,
        split="dev",
        seed=42,
        tokenizer=handle.tokenizer
    )
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    IO_token_ids = [p.target_token_id for p in pairs]
    S_token_ids = [p.foil_token_id for p in pairs]
    
    print(f"Loaded {len(pairs)} clean/corrupt pairs")
    
    # Cache clean head z activations for all layers
    print("\n[3/5] Caching clean head z activations (all layers)...")
    clean_cache = cache_head_z(handle, clean_prompts)
    print(f"Cached layers: {list(clean_cache.keys())}")
    
    # Define metric function
    def metric_fn(logits):
        """Compute logit_diff for each sample in the batch."""
        return logit_diff(logits, IO_token_ids, S_token_ids)
    
    # Run head_patch_sweep
    print("\n[4/5] Running head_patch_sweep (this may take a few minutes)...")
    print(f"Sweeping {handle.n_layers} layers × {handle.n_heads} heads = {handle.n_layers * handle.n_heads} sites")
    
    sweep = head_patch_sweep(
        handle,
        clean_prompts,
        corrupt_prompts,
        metric_fn,
        patch_positions=[-1],
        clean_cache=clean_cache
    )
    
    recovery_matrix = sweep["recovery"].cpu().numpy()  # shape: [n_layers, n_heads]
    print(f"Recovery matrix shape: {recovery_matrix.shape}")
    
    # Find top-K heads
    print("\n[5/5] Analyzing results...")
    
    # Flatten and rank
    flat_recovery = []
    for layer in range(recovery_matrix.shape[0]):
        for head in range(recovery_matrix.shape[1]):
            flat_recovery.append({
                "layer": layer,
                "head": head,
                "recovery": float(recovery_matrix[layer, head])
            })
    
    # Sort by recovery (descending)
    flat_recovery.sort(key=lambda x: x["recovery"], reverse=True)
    
    # Print top-10
    print(f"\nTop-10 heads by recovery:")
    for i, site in enumerate(flat_recovery[:10]):
        print(f"  {i+1}. L{site['layer']}H{site['head']}: {site['recovery']:.4f}")
    
    # Get top-K sites for joint patch
    top_K = 3
    top_sites_sweep = [(site['layer'], site['head']) for site in flat_recovery[:top_K]]
    print(f"\nTop-{top_K} heads from sweep: {top_sites_sweep}")
    
    # Compare with discovery results
    discovery_sites = [(9, 9), (9, 6), (10, 0)]
    print(f"Top-3 heads from discovery: {discovery_sites}")
    
    agreement = set(top_sites_sweep) == set(discovery_sites)
    print(f"Discovery-Localization agreement: {agreement}")
    
    # Save results
    results = {
        "recovery_matrix": recovery_matrix.tolist(),
        "top_10_heads": flat_recovery[:10],
        "top_3_heads_sweep": top_sites_sweep,
        "top_3_heads_discovery": discovery_sites,
        "agreement": agreement,
        "n_layers": handle.n_layers,
        "n_heads": handle.n_heads,
        "n_samples": len(clean_prompts)
    }
    
    output_path = Path("runs/ioi-discovery-v1_rev3/scratch/stage2_head_sweep_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save recovery matrix as numpy array for easier inspection
    np.save("runs/ioi-discovery-v1_rev3/scratch/stage2_recovery_matrix.npy", recovery_matrix)
    
    print(f"\n{'='*80}")
    print(f"HEAD_PATCH_SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to:")
    print(f"  - {output_path}")
    print(f"  - runs/ioi-discovery-v1_rev3/scratch/stage2_recovery_matrix.npy")
    print(f"\nNext: Run joint top-K patch on sweep results and compute metrics")

if __name__ == "__main__":
    main()
