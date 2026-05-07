#!/usr/bin/env python3
"""Stage 2B: Joint patch with localization top-K heads."""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    cache_head_z, HeadPatch, run_with_head_patches
)

def main():
    run_dir = Path(__file__).parent.parent
    spec_path = run_dir / "spec.json"
    
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load sweep results
    with open(run_dir / "scratch" / "stage2_head_sweep_results.json") as f:
        sweep = json.load(f)
    
    # Extract top heads from sweep
    loc_top_10 = [(h['layer'], h['head']) for h in sweep['top_20'][:10]]
    
    print("Localization top-10 heads:")
    for i, (l, h) in enumerate(loc_top_10):
        rec = [x for x in sweep['top_20'] if x['layer']==l and x['head']==h][0]['recovery']
        print(f"  {i+1}. L{l}H{h}: {rec:.4f}")
    
    # Load model and data
    print("\nLoading model...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    pairs = generate_pairs(
        spec, n_pairs=spec['dataset']['n_samples'],
        seed=spec['dataset']['seed'], tokenizer=handle.tokenizer, split="dev"
    )
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    IO_token_ids = torch.tensor([p.target_token_id for p in pairs], device=handle.device)
    S_token_ids = torch.tensor([p.foil_token_id for p in pairs], device=handle.device)
    
    # Compute baseline
    def compute_logit_diff(prompts, io_ids, s_ids):
        with torch.no_grad():
            inputs = handle.tokenizer(prompts, return_tensors="pt", padding=True).to(handle.device)
            logits = handle.model(**inputs).logits[:, -1, :]
            io_logits = logits[torch.arange(len(prompts)), io_ids]
            s_logits = logits[torch.arange(len(prompts)), s_ids]
            return (io_logits - s_logits).mean().item()
    
    clean_lD = compute_logit_diff(clean_prompts, IO_token_ids, S_token_ids)
    corrupt_lD = compute_logit_diff(corrupt_prompts, IO_token_ids, S_token_ids)
    gap = clean_lD - corrupt_lD
    
    print(f"\nBaseline:")
    print(f"  Clean:   {clean_lD:.4f}")
    print(f"  Corrupt: {corrupt_lD:.4f}")
    print(f"  Gap:     {gap:.4f}")
    
    # Try different K values from localization top-K
    print("\n=== Joint patches with localization top-K ===")
    results = []
    
    for K in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        top_K = loc_top_10[:K]
        layers = sorted({L for L, H in top_K})
        clean_z = cache_head_z(handle, clean_prompts, layers=layers)
        
        patches = [HeadPatch(layer=L, head=H, source=clean_z[L][:, :, H, :]) for (L, H) in top_K]
        patched_logits = run_with_head_patches(handle, corrupt_prompts, patches, return_logits_at=-1)
        
        joint_io_logits = patched_logits[torch.arange(len(pairs)), IO_token_ids]
        joint_s_logits = patched_logits[torch.arange(len(pairs)), S_token_ids]
        joint_lD = (joint_io_logits - joint_s_logits).mean().item()
        
        recovery = (joint_lD - corrupt_lD) / gap
        
        print(f"K={K:2d}: recovery={recovery:.4f}")
        
        results.append({
            "K": K,
            "top_K": top_K,
            "joint_logit_diff": joint_lD,
            "recovery": recovery
        })
    
    # Find best K >= threshold (0.85 for localization-recovery criterion)
    best = max(results, key=lambda x: x['recovery'])
    best_over_threshold = [r for r in results if r['recovery'] >= 0.85]
    
    print(f"\n=== BEST ===")
    print(f"K={best['K']}: recovery={best['recovery']:.4f}")
    
    if best_over_threshold:
        best_thresh = best_over_threshold[0]
        print(f"\nBest meeting 0.85 threshold:")
        print(f"K={best_thresh['K']}: recovery={best_thresh['recovery']:.4f}")
    else:
        print(f"\nNO K meets 0.85 threshold! Best is {best['recovery']:.4f}")
    
    # Save
    output = {
        "baseline": {"clean_lD": clean_lD, "corrupt_lD": corrupt_lD, "gap": gap},
        "K_sweep": results,
        "best": best,
        "best_over_threshold": best_over_threshold[0] if best_over_threshold else None
    }
    
    with open(run_dir / "scratch" / "stage2_joint_patch_localization.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to scratch/stage2_joint_patch_localization.json")

if __name__ == "__main__":
    main()
