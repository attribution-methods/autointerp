#!/usr/bin/env python3
"""
Stage 1: Joint top-K patch on dev split.
Extract top-K heads from discovery and run joint simultaneous patch.
"""

import sys
sys.path.insert(0, "src")

import torch
import json
from pathlib import Path

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    HeadPatch, cache_head_z, run_with_head_patches, logit_diff
)

def main():
    print("=" * 80)
    print("STAGE 1: JOINT TOP-K PATCH ON DEV")
    print("=" * 80)
    
    # Load spec
    spec_path = Path("runs/ioi-discovery-v1_rev3/spec.json")
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load model
    print("\n[1/7] Loading GPT-2-small...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded: {handle.model_id}, device={handle.device}")
    
    # Load IOI dataset
    print("\n[2/7] Loading IOI dev dataset (n=500, seed=42)...")
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
    
    # Extract top-K heads from discovery results
    print("\n[3/7] Extracting top-K heads from discovery...")
    discovery_path = Path("runs/ioi-discovery-v1_rev3/discovery/stage01_call5/results/algorithm_v3/harness_result.json")
    with open(discovery_path) as f:
        discovery_result = json.load(f)
    
    # Get top-3 heads with positive scores
    all_features = discovery_result['top_features']
    positive_features = [f for f in all_features if f['score'] > 0]
    top_K = 3
    top_features = positive_features[:top_K]
    
    top_sites = [(f['layer'], f['idx']) for f in top_features]
    print(f"Top-{top_K} heads (positive scores):")
    for f in top_features:
        print(f"  L{f['layer']}H{f['idx']}: score={f['score']:.4f}")
    
    # Compute clean and corrupt logit_diff baselines
    print("\n[4/7] Computing clean baseline...")
    clean_logits_list = []
    batch_size = 32
    
    for i in range(0, len(clean_prompts), batch_size):
        batch = clean_prompts[i:i+batch_size]
        inputs = handle.tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(handle.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = handle.model(**inputs)
            logits = outputs.logits
            last_positions = inputs['attention_mask'].sum(dim=1) - 1
            batch_logits = torch.stack([
                logits[j, last_positions[j], :]
                for j in range(len(batch))
            ])
            clean_logits_list.append(batch_logits.cpu())
    
    clean_logits = torch.cat(clean_logits_list, dim=0)
    clean_lD = logit_diff(clean_logits, IO_token_ids, S_token_ids).mean().item()
    print(f"Clean logit_diff: {clean_lD:.4f}")
    
    print("\n[5/7] Computing corrupt baseline...")
    corrupt_logits_list = []
    
    for i in range(0, len(corrupt_prompts), batch_size):
        batch = corrupt_prompts[i:i+batch_size]
        inputs = handle.tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(handle.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = handle.model(**inputs)
            logits = outputs.logits
            last_positions = inputs['attention_mask'].sum(dim=1) - 1
            batch_logits = torch.stack([
                logits[j, last_positions[j], :]
                for j in range(len(batch))
            ])
            corrupt_logits_list.append(batch_logits.cpu())
    
    corrupt_logits = torch.cat(corrupt_logits_list, dim=0)
    corrupt_lD = logit_diff(corrupt_logits, IO_token_ids, S_token_ids).mean().item()
    print(f"Corrupt logit_diff: {corrupt_lD:.4f}")
    print(f"Clean-corrupt gap: {clean_lD - corrupt_lD:.4f}")
    
    # Cache clean head z activations
    print("\n[6/7] Caching clean head z activations...")
    layers_needed = sorted({L for (L, H) in top_sites})
    print(f"Layers needed: {layers_needed}")
    
    clean_cache = cache_head_z(handle, clean_prompts, layers=layers_needed)
    print(f"Cached layers: {list(clean_cache.keys())}")
    
    # Run joint patch: patch all top-K heads simultaneously
    print("\n[7/7] Running joint top-K patch...")
    patches = [
        HeadPatch(layer=L, head=H, source=clean_cache[L][:, :, H, :])
        for (L, H) in top_sites
    ]
    print(f"Created {len(patches)} patches for joint application")
    
    patched_logits = run_with_head_patches(
        handle, corrupt_prompts, patches, return_logits_at=-1
    )
    
    joint_lD = logit_diff(patched_logits, IO_token_ids, S_token_ids).mean().item()
    recovery = (joint_lD - corrupt_lD) / (clean_lD - corrupt_lD)
    
    print(f"\n{'='*80}")
    print(f"JOINT TOP-{top_K} PATCH RESULTS")
    print(f"{'='*80}")
    print(f"Heads: {top_sites}")
    print(f"Clean logit_diff:   {clean_lD:.4f}")
    print(f"Corrupt logit_diff: {corrupt_lD:.4f}")
    print(f"Patched logit_diff: {joint_lD:.4f}")
    print(f"Patch effect recovery: {recovery:.4f}")
    print(f"{'='*80}")
    
    # Save results
    results = {
        "top_sites": top_sites,
        "top_features": top_features,
        "clean_logit_diff": clean_lD,
        "corrupt_logit_diff": corrupt_lD,
        "patched_logit_diff": joint_lD,
        "patch_effect_recovery": recovery,
        "n_samples": len(clean_prompts)
    }
    
    output_path = Path("runs/ioi-discovery-v1_rev3/scratch/stage1_joint_patch_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nNext: Commit patch_effect_recovery MetricResult and evaluate discovery-recovery criterion")

if __name__ == "__main__":
    main()
