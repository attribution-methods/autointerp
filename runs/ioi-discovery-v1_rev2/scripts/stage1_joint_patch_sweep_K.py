#!/usr/bin/env python3
"""Sweep different K values to find best joint recovery."""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    HeadPatch, cache_head_z, run_with_head_patches
)

def main():
    run_dir = Path(__file__).parent.parent
    spec_path = run_dir / "spec.json"
    
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load discovery results
    discovery_result_path = run_dir / "discovery" / "stage01_call6" / "results" / "algorithm_v4" / "harness_result.json"
    with open(discovery_result_path) as f:
        discovery_result = json.load(f)
    
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
    
    # Compute baseline logit_diffs
    def compute_logit_diff(prompts, io_ids, s_ids):
        with torch.no_grad():
            inputs = handle.tokenizer(prompts, return_tensors="pt", padding=True).to(handle.device)
            logits = handle.model(**inputs).logits[:, -1, :]
            io_logits = logits[torch.arange(len(prompts)), io_ids]
            s_logits = logits[torch.arange(len(prompts)), s_ids]
            return (io_logits - s_logits).mean().item()
    
    print("Computing baseline logit_diffs...")
    clean_lD = compute_logit_diff(clean_prompts, IO_token_ids, S_token_ids)
    corrupt_lD = compute_logit_diff(corrupt_prompts, IO_token_ids, S_token_ids)
    gap = clean_lD - corrupt_lD
    
    print(f"\nBaseline:")
    print(f"  Clean:   {clean_lD:.4f}")
    print(f"  Corrupt: {corrupt_lD:.4f}")
    print(f"  Gap:     {gap:.4f}")
    
    # Try different K values
    print("\n=== Sweeping K values ===")
    results = []
    
    for K in [3, 4, 5, 6, 7, 8]:
        top_features = discovery_result['top_features'][:K]
        top_K = [(f['layer'], f['idx']) for f in top_features]
        
        # Cache head z for these layers
        layers = sorted({L for L, H in top_K})
        clean_z = cache_head_z(handle, clean_prompts, layers=layers)
        
        # Build patches
        patches = [HeadPatch(layer=L, head=H, source=clean_z[L][:, :, H, :]) for (L, H) in top_K]
        
        # Run joint patch
        patched_logits = run_with_head_patches(handle, corrupt_prompts, patches, return_logits_at=-1)
        joint_io_logits = patched_logits[torch.arange(len(pairs)), IO_token_ids]
        joint_s_logits = patched_logits[torch.arange(len(pairs)), S_token_ids]
        joint_lD = (joint_io_logits - joint_s_logits).mean().item()
        
        recovery = (joint_lD - corrupt_lD) / gap
        
        print(f"\nK={K}: recovery={recovery:.4f}")
        print(f"  Heads: {[(L, H) for L, H in top_K]}")
        
        results.append({
            "K": K,
            "top_K": top_K,
            "joint_logit_diff": joint_lD,
            "recovery": recovery
        })
    
    # Save results
    output = {
        "baseline": {"clean_lD": clean_lD, "corrupt_lD": corrupt_lD, "gap": gap},
        "K_sweep": results
    }
    output_path = run_dir / "scratch" / "stage1_K_sweep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Find best K
    best = max(results, key=lambda x: x['recovery'])
    print(f"\n=== BEST ===")
    print(f"K={best['K']}: recovery={best['recovery']:.4f}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
