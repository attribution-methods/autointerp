#!/usr/bin/env python3
"""Stage 1B: Joint top-K patch on dev split.

After discovery, this script:
1. Loads the top-K (layer, head) sites from discovery results
2. Generates clean and corrupted IOI pairs for dev split
3. Computes baseline logit_diffs (clean, corrupt)
4. Caches head z-activations on clean prompts
5. Runs a JOINT patch: all K heads patched simultaneously from clean to corrupt
6. Computes joint patch_effect_recovery
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    HeadPatch, cache_head_z, run_with_head_patches, logit_diff
)

def main():
    run_dir = Path(__file__).parent.parent
    spec_path = run_dir / "spec.json"
    
    # Load spec
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load discovery results
    discovery_result_path = run_dir / "discovery" / "stage01_call6" / "results" / "algorithm_v4" / "harness_result.json"
    with open(discovery_result_path) as f:
        discovery_result = json.load(f)
    
    # Extract top-K sites (K=3)
    K = 3
    top_features = discovery_result['top_features'][:K]
    top_K = [(f['layer'], f['idx']) for f in top_features]
    
    print(f"Top-{K} heads from discovery:")
    for i, (layer, head) in enumerate(top_K):
        score = top_features[i]['score']
        print(f"  {i+1}. Layer {layer}, Head {head} (score={score})")
    
    # Load model
    print("\nLoading GPT-2-small model...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate IOI pairs (dev split, same as stage 0)
    print(f"\nGenerating {spec['dataset']['n_samples']} IOI pairs for dev split...")
    pairs = generate_pairs(
        spec,
        n_pairs=spec['dataset']['n_samples'],
        seed=spec['dataset']['seed'],
        tokenizer=handle.tokenizer,
        split="dev"
    )
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    IO_token_ids = torch.tensor([p.target_token_id for p in pairs], device=handle.device)
    S_token_ids = torch.tensor([p.foil_token_id for p in pairs], device=handle.device)
    
    print(f"Generated {len(pairs)} pairs")
    print(f"  Sample clean:   '{clean_prompts[0]}'")
    print(f"  Sample corrupt: '{corrupt_prompts[0]}'")
    
    # Compute baseline logit_diffs
    print("\n=== Computing baseline logit_diffs ===")
    
    def compute_logit_diff(prompts, io_ids, s_ids, label):
        """Compute mean logit_diff for a batch of prompts."""
        with torch.no_grad():
            inputs = handle.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(handle.device)
            
            outputs = handle.model(**inputs)
            logits = outputs.logits[:, -1, :]  # [B, vocab]
            
            # Extract logits for IO and S tokens
            io_logits = logits[torch.arange(len(prompts)), io_ids]
            s_logits = logits[torch.arange(len(prompts)), s_ids]
            
            return (io_logits - s_logits).mean().item()
    
    clean_lD = compute_logit_diff(clean_prompts, IO_token_ids, S_token_ids, "clean")
    corrupt_lD = compute_logit_diff(corrupt_prompts, IO_token_ids, S_token_ids, "corrupt")
    
    print(f"Clean logit_diff:   {clean_lD:.4f}")
    print(f"Corrupt logit_diff: {corrupt_lD:.4f}")
    print(f"Gap (clean - corrupt): {clean_lD - corrupt_lD:.4f}")
    
    # Cache head z-activations on clean prompts
    print(f"\n=== Caching head z on clean prompts for layers {sorted({L for L, H in top_K})} ===")
    clean_z = cache_head_z(handle, clean_prompts, layers=sorted({L for L, H in top_K}))
    
    for layer in sorted({L for L, H in top_K}):
        print(f"  Layer {layer}: shape {clean_z[layer].shape}")
    
    # Build HeadPatch objects for joint patching
    patches = [
        HeadPatch(layer=L, head=H, source=clean_z[L][:, :, H, :])
        for (L, H) in top_K
    ]
    
    print(f"\n=== Running JOINT patch: all {K} heads simultaneously ===")
    patched_logits = run_with_head_patches(
        handle,
        corrupt_prompts,
        patches,
        return_logits_at=-1
    )  # [B, vocab]
    
    # Compute joint logit_diff
    joint_io_logits = patched_logits[torch.arange(len(pairs)), IO_token_ids]
    joint_s_logits = patched_logits[torch.arange(len(pairs)), S_token_ids]
    joint_lD = (joint_io_logits - joint_s_logits).mean().item()
    
    # Compute recovery
    recovery = (joint_lD - corrupt_lD) / (clean_lD - corrupt_lD)
    
    print(f"\nJoint patched logit_diff: {joint_lD:.4f}")
    print(f"Recovery: ({joint_lD:.4f} - {corrupt_lD:.4f}) / ({clean_lD:.4f} - {corrupt_lD:.4f}) = {recovery:.4f}")
    
    # Save results
    results = {
        "top_K": top_K,
        "K": K,
        "clean_logit_diff": clean_lD,
        "corrupt_logit_diff": corrupt_lD,
        "gap": clean_lD - corrupt_lD,
        "joint_patched_logit_diff": joint_lD,
        "patch_effect_recovery": recovery,
        "n_samples": len(pairs)
    }
    
    output_path = run_dir / "scratch" / "stage1_joint_patch_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print(f"Top-{K} heads: {top_K}")
    print(f"Joint patch_effect_recovery: {recovery:.4f}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
