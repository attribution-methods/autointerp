#!/usr/bin/env python3
"""Stage 2: Exhaustive head patch sweep for localization.

Runs head_patch_sweep across all 144 (layer, head) pairs in GPT-2-small,
then identifies top-K by recovery and runs a joint patch.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    cache_head_z, head_patch_sweep, HeadPatch, run_with_head_patches, logit_diff
)

def main():
    run_dir = Path(__file__).parent.parent
    spec_path = run_dir / "spec.json"
    
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load model
    print("Loading GPT-2-small model...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate IOI pairs
    print("Generating IOI pairs...")
    pairs = generate_pairs(
        spec, n_pairs=spec['dataset']['n_samples'],
        seed=spec['dataset']['seed'], tokenizer=handle.tokenizer, split="dev"
    )
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    IO_token_ids = torch.tensor([p.target_token_id for p in pairs], device=handle.device)
    S_token_ids = torch.tensor([p.foil_token_id for p in pairs], device=handle.device)
    
    print(f"Generated {len(pairs)} pairs")
    
    # Define metric
    def metric_fn(logits):
        """Logit diff at final position: logit(IO) - logit(S)."""
        return logit_diff(logits, IO_token_ids, S_token_ids)
    
    # Cache clean head z
    print("\n=== Caching clean head z for all layers ===")
    clean_cache = cache_head_z(handle, clean_prompts)
    
    print(f"Cached {len(clean_cache)} layers")
    for layer in sorted(clean_cache.keys())[:3]:
        print(f"  Layer {layer}: {clean_cache[layer].shape}")
    print("  ...")
    
    # Run exhaustive head patch sweep
    print("\n=== Running head_patch_sweep across all 144 (layer, head) pairs ===")
    print("This will take several minutes...")
    
    sweep_result = head_patch_sweep(
        handle,
        clean_prompts,
        corrupt_prompts,
        metric_fn,
        patch_positions=[-1],
        clean_cache=clean_cache
    )
    
    recovery_matrix = sweep_result["recovery"]  # [L, H]
    print(f"\nSweep complete. Recovery matrix shape: {recovery_matrix.shape}")
    print(f"  Mean recovery: {recovery_matrix.mean().item():.4f}")
    print(f"  Max recovery: {recovery_matrix.max().item():.4f}")
    print(f"  Min recovery: {recovery_matrix.min().item():.4f}")
    
    # Find top-K heads by recovery
    n_layers, n_heads = recovery_matrix.shape
    flat_recoveries = recovery_matrix.flatten()
    flat_indices = torch.argsort(flat_recoveries, descending=True)
    
    # Print top 20
    print("\n=== Top 20 heads by recovery ===")
    top_20_sites = []
    for rank in range(20):
        idx = flat_indices[rank].item()
        layer = idx // n_heads
        head = idx % n_heads
        recovery = flat_recoveries[idx].item()
        top_20_sites.append((layer, head, recovery))
        print(f"{rank+1:2d}. L{layer:2d}H{head:2d}: {recovery:.4f}")
    
    # Save full results
    results = {
        "recovery_matrix": recovery_matrix.cpu().numpy().tolist(),
        "shape": list(recovery_matrix.shape),
        "top_20": [
            {"rank": i+1, "layer": l, "head": h, "recovery": r}
            for i, (l, h, r) in enumerate(top_20_sites)
        ],
        "mean_recovery": recovery_matrix.mean().item(),
        "max_recovery": recovery_matrix.max().item(),
        "min_recovery": recovery_matrix.min().item()
    }
    
    output_path = run_dir / "scratch" / "stage2_head_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSweep results saved to {output_path}")
    
    # Compare with discovery top-5
    discovery_top_5 = [(9, 9), (10, 7), (9, 6), (11, 10), (10, 0)]
    print("\n=== Comparison with discovery top-5 ===")
    print("Discovery top-5 heads and their localization recovery:")
    for i, (l, h) in enumerate(discovery_top_5):
        recovery = recovery_matrix[l, h].item()
        # Find rank in localization
        loc_rank = (flat_recoveries >= recovery).sum().item()
        print(f"  {i+1}. L{l}H{h}: recovery={recovery:.4f}, localization_rank={loc_rank}")
    
    # Check agreement: how many of discovery top-5 are in localization top-10?
    loc_top_10 = set((flat_indices[i].item() // n_heads, flat_indices[i].item() % n_heads) 
                     for i in range(10))
    discovery_set = set(discovery_top_5)
    overlap = discovery_set.intersection(loc_top_10)
    
    print(f"\nOverlap: {len(overlap)}/5 discovery heads are in localization top-10")
    print(f"  In common: {sorted(overlap)}")
    print(f"  Discovery only: {sorted(discovery_set - overlap)}")
    print(f"  Localization only: {sorted(loc_top_10 - discovery_set)}")

if __name__ == "__main__":
    main()
