#!/usr/bin/env python3
"""
Stage 2: Joint top-K patch and compute kl_to_clean and patch_effect_recovery.
"""

import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import json
from pathlib import Path

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    HeadPatch, cache_head_z, run_with_head_patches, logit_diff
)

def compute_kl_divergence(logits1, logits2, return_per_sample=False):
    """Compute KL divergence between two logit distributions.
    
    Args:
        logits1: Clean logits (B, V)
        logits2: Patched logits (B, V)
        return_per_sample: If True, return per-sample KL values
        
    Returns:
        If return_per_sample=False: scalar mean KL
        If return_per_sample=True: (mean_kl, per_sample_kl_list)
    """
    # Ensure both tensors are on the same device
    device = logits1.device if hasattr(logits1, 'device') else 'cpu'
    logits1 = logits1.to(device) if hasattr(logits1, 'to') else torch.tensor(logits1).to(device)
    logits2 = logits2.to(device) if hasattr(logits2, 'to') else torch.tensor(logits2).to(device)
    
    # Convert logits to log probabilities
    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    
    # KL(p2 || p1) = sum(p2 * (log_p2 - log_p1))
    kl_per_sample = (p2 * (log_p2 - log_p1)).sum(dim=-1)  # (B,)
    
    if return_per_sample:
        return kl_per_sample.mean().item(), kl_per_sample.cpu().tolist()
    else:
        return kl_per_sample.mean().item()

def main():
    print("=" * 80)
    print("STAGE 2: JOINT PATCH AND METRICS")
    print("=" * 80)
    
    # Load spec
    spec_path = Path("runs/ioi-discovery-v1_rev3/spec.json")
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load model
    print("\n[1/8] Loading GPT-2-small...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded: {handle.model_id}")
    
    # Load IOI dataset
    print("\n[2/8] Loading IOI dev dataset (n=500, seed=42)...")
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
    
    # Load sweep results to get top-K heads
    print("\n[3/8] Loading head sweep results...")
    sweep_path = Path("runs/ioi-discovery-v1_rev3/scratch/stage2_head_sweep_results.json")
    with open(sweep_path) as f:
        sweep_results = json.load(f)
    
    top_sites = [(h['layer'], h['head']) for h in sweep_results['top_10_heads'][:3]]
    print(f"Top-3 heads from sweep: {top_sites}")
    
    # Compute clean baseline
    print("\n[4/8] Computing clean baseline logits...")
    batch_size = 32
    clean_logits_list = []
    
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
    
    # Compute corrupt baseline
    print("\n[5/8] Computing corrupt baseline logits...")
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
    print("\n[6/8] Caching clean head z activations...")
    layers_needed = sorted({L for (L, H) in top_sites})
    clean_cache = cache_head_z(handle, clean_prompts, layers=layers_needed)
    print(f"Cached layers: {list(clean_cache.keys())}")
    
    # Run joint patch
    print("\n[7/8] Running joint top-K patch...")
    patches = [
        HeadPatch(layer=L, head=H, source=clean_cache[L][:, :, H, :])
        for (L, H) in top_sites
    ]
    
    patched_logits = run_with_head_patches(
        handle, corrupt_prompts, patches, return_logits_at=-1
    )
    
    joint_lD = logit_diff(patched_logits, IO_token_ids, S_token_ids).mean().item()
    recovery = (joint_lD - corrupt_lD) / (clean_lD - corrupt_lD)
    
    print(f"Patched logit_diff: {joint_lD:.4f}")
    print(f"Patch effect recovery: {recovery:.4f}")
    
    # Compute KL divergence
    print("\n[8/8] Computing KL divergence...")
    kl_mean, kl_per_sample = compute_kl_divergence(clean_logits, patched_logits, return_per_sample=True)
    print(f"KL(clean || patched): {kl_mean:.4f}")
    
    print(f"\n{'='*80}")
    print(f"STAGE 2 METRICS")
    print(f"{'='*80}")
    print(f"Top-3 heads: {top_sites}")
    print(f"Clean logit_diff:   {clean_lD:.4f}")
    print(f"Corrupt logit_diff: {corrupt_lD:.4f}")
    print(f"Patched logit_diff: {joint_lD:.4f}")
    print(f"Patch effect recovery: {recovery:.4f}")
    print(f"KL(clean || patched): {kl_mean:.4f}")
    print(f"{'='*80}")
    
    # Save results
    results = {
        "top_sites": top_sites,
        "clean_logit_diff": clean_lD,
        "corrupt_logit_diff": corrupt_lD,
        "patched_logit_diff": joint_lD,
        "patch_effect_recovery": recovery,
        "kl_to_clean_mean": kl_mean,
        "kl_per_sample": kl_per_sample,
        "n_samples": len(clean_prompts)
    }
    
    output_path = Path("runs/ioi-discovery-v1_rev3/scratch/stage2_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nNext: Commit kl_to_clean and patch_effect_recovery MetricResults")

if __name__ == "__main__":
    main()
