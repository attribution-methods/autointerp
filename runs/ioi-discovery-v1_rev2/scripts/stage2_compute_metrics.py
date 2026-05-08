#!/usr/bin/env python3
"""Compute kl_to_clean and patch_effect_recovery for stage 2."""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import cache_head_z, HeadPatch, run_with_head_patches

def main():
    run_dir = Path(__file__).parent.parent
    spec_path = run_dir / "spec.json"
    
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load localization results
    with open(run_dir / "scratch" / "stage2_joint_patch_localization.json") as f:
        loc_results = json.load(f)
    
    # Get K=3 top heads (the best meeting threshold)
    K = 3
    top_K = [(9, 9), (9, 6), (10, 0)]  # From localization sweep
    
    print(f"Computing metrics for K={K} joint patch:")
    for i, (l, h) in enumerate(top_K):
        print(f"  {i+1}. L{l}H{h}")
    
    # Load model and data
    print("\nLoading model and data...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    pairs = generate_pairs(
        spec, n_pairs=spec['dataset']['n_samples'],
        seed=spec['dataset']['seed'], tokenizer=handle.tokenizer, split="dev"
    )
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    IO_token_ids = torch.tensor([p.target_token_id for p in pairs], device=handle.device)
    S_token_ids = torch.tensor([p.foil_token_id for p in pairs], device=handle.device)
    
    # Compute clean logits
    print("Computing clean logits...")
    with torch.no_grad():
        clean_inputs = handle.tokenizer(clean_prompts, return_tensors="pt", padding=True).to(handle.device)
        clean_outputs = handle.model(**clean_inputs)
        clean_logits = clean_outputs.logits[:, -1, :]  # [B, vocab]
    
    # Compute corrupt logits
    print("Computing corrupt logits...")
    with torch.no_grad():
        corrupt_inputs = handle.tokenizer(corrupt_prompts, return_tensors="pt", padding=True).to(handle.device)
        corrupt_outputs = handle.model(**corrupt_inputs)
        corrupt_logits = corrupt_outputs.logits[:, -1, :]
    
    # Compute patched logits (K=3 joint)
    print("Computing patched logits with K=3 joint patch...")
    layers = sorted({L for L, H in top_K})
    clean_z = cache_head_z(handle, clean_prompts, layers=layers)
    patches = [HeadPatch(layer=L, head=H, source=clean_z[L][:, :, H, :]) for (L, H) in top_K]
    patched_logits = run_with_head_patches(handle, corrupt_prompts, patches, return_logits_at=-1)
    
    # Compute logit_diffs
    clean_io = clean_logits[torch.arange(len(pairs)), IO_token_ids]
    clean_s = clean_logits[torch.arange(len(pairs)), S_token_ids]
    clean_lD = (clean_io - clean_s).mean().item()
    
    corrupt_io = corrupt_logits[torch.arange(len(pairs)), IO_token_ids]
    corrupt_s = corrupt_logits[torch.arange(len(pairs)), S_token_ids]
    corrupt_lD = (corrupt_io - corrupt_s).mean().item()
    
    patched_io = patched_logits[torch.arange(len(pairs)), IO_token_ids]
    patched_s = patched_logits[torch.arange(len(pairs)), S_token_ids]
    patched_lD = (patched_io - patched_s).mean().item()
    
    # Compute patch_effect_recovery
    gap = clean_lD - corrupt_lD
    recovery = (patched_lD - corrupt_lD) / gap
    
    print(f"\nLogit diffs:")
    print(f"  Clean:   {clean_lD:.4f}")
    print(f"  Corrupt: {corrupt_lD:.4f}")
    print(f"  Patched: {patched_lD:.4f}")
    print(f"  Gap:     {gap:.4f}")
    print(f"  Recovery: {recovery:.4f}")
    
    # Compute KL(patched || clean) per sample
    print("\nComputing KL divergence per sample...")
    clean_probs = F.softmax(clean_logits, dim=-1)  # [B, vocab]
    patched_probs = F.softmax(patched_logits, dim=-1)
    
    # KL(patched || clean) = sum_i patched_probs[i] * log(patched_probs[i] / clean_probs[i])
    kl_per_sample = F.kl_div(
        clean_probs.log(), patched_probs,
        reduction='none', log_target=False
    ).sum(dim=-1).cpu().tolist()  # [B]
    
    mean_kl = sum(kl_per_sample) / len(kl_per_sample)
    
    print(f"  Mean KL(patched || clean): {mean_kl:.4f}")
    print(f"  Min KL: {min(kl_per_sample):.4f}")
    print(f"  Max KL: {max(kl_per_sample):.4f}")
    
    # Save metrics
    metrics = {
        "top_K": top_K,
        "K": K,
        "patch_effect_recovery": {
            "clean_metric": clean_lD,
            "corrupt_metric": corrupt_lD,
            "patched_metric": patched_lD,
            "value": recovery
        },
        "kl_to_clean": {
            "kl_per_sample": kl_per_sample,
            "mean_kl": mean_kl,
            "min_kl": min(kl_per_sample),
            "max_kl": max(kl_per_sample)
        }
    }
    
    output_path = run_dir / "scratch" / "stage2_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_path}")
    
    # Print summary for compute_metric calls
    print("\n=== For compute_metric calls ===")
    print(f"patch_effect_recovery inputs:")
    print(f"  clean_metric: {clean_lD}")
    print(f"  corrupt_metric: {corrupt_lD}")
    print(f"  patched_metric: {patched_lD}")
    print(f"\nkl_to_clean inputs:")
    print(f"  kl_per_sample: list of {len(kl_per_sample)} values (mean={mean_kl:.4f})")

if __name__ == "__main__":
    main()
