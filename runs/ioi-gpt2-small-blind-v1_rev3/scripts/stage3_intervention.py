"""Stage 3: Intervention - path patching and faithfulness on dev.

Measure circuit faithfulness using END-ONLY ablation methodology.
"""
import sys
sys.path.insert(0, "src")

import torch
import json
from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    cache_head_z, mean_head_z, run_with_head_patches, HeadPatch, logit_diff
)

def main():
    # Load model
    print("Loading model gpt2...")
    handle = load_model("gpt2")
    
    # Generate IOI pairs
    print("\nGenerating 500 IOI dev pairs...")
    pairs = generate_pairs(None, n_pairs=500, seed=42, tokenizer=handle.tokenizer, split="dev")
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    io_ids = torch.tensor([p.target_token_id for p in pairs])
    s_ids = torch.tensor([p.foil_token_id for p in pairs])
    
    # Load circuit from stage1
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage1_results.json") as f:
        stage1 = json.load(f)
    
    circuit_heads = [(L, H) for L, H in stage1["top_K_heads"]]
    print(f"\nCircuit C = {circuit_heads}")
    
    # Compute baselines
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
    
    print(f"Clean logit_diff: {clean_lD:.4f}")
    print(f"Corrupt logit_diff: {corrupt_lD:.4f}")
    print(f"Gap: {clean_lD - corrupt_lD:.4f}")
    
    # Measure circuit faithfulness with END-ONLY ablation
    print("\n=== Measuring circuit faithfulness (END-ONLY ablation) ===")
    
    # Cache mean head z over clean batch (broadcast baseline for ablation)
    clean_cache = cache_head_z(handle, clean_prompts)
    mean_z = mean_head_z(clean_cache)  # {layer: [seq, n_heads, d_head]} averaged over batch
    
    # Build patches for ALL heads NOT in circuit C, at END position ONLY
    circuit_set = set(circuit_heads)
    patches = []
    
    for L in range(handle.n_layers):
        for H in range(handle.n_heads):
            if (L, H) not in circuit_set:
                # mean_z[L] shape: [seq, n_heads, d_head]
                # Extract head H: [seq, d_head]
                source_z = mean_z[L][:, H, :]
                # Apply only at END position
                patches.append(HeadPatch(layer=L, head=H, source=source_z, positions=[-1]))
    
    print(f"Ablating {len(patches)} heads (complement of C) at END position only...")
    
    # Run circuit-only forward (complement ablated at END)
    circuit_logits = run_with_head_patches(handle, clean_prompts, patches, return_logits_at=-1)
    circuit_lD = logit_diff(circuit_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
    
    # Compute faithfulness
    faithfulness_value = (circuit_lD - corrupt_lD) / (clean_lD - corrupt_lD)
    
    print(f"\nCircuit-only logit_diff: {circuit_lD:.4f}")
    print(f"Faithfulness: {faithfulness_value:.4f}")
    print(f"  = (circuit - corrupt) / (clean - corrupt)")
    print(f"  = ({circuit_lD:.4f} - {corrupt_lD:.4f}) / ({clean_lD:.4f} - {corrupt_lD:.4f})")
    
    # Also compute patch_effect_recovery for the circuit (same as stage1 joint recovery)
    # This is already computed in stage1, but let's recompute for completeness
    clean_cache_full = cache_head_z(handle, clean_prompts)
    circuit_patches = []
    for L, H in circuit_heads:
        source_z = clean_cache_full[L][:, :, H, :]
        circuit_patches.append(HeadPatch(layer=L, head=H, source=source_z, positions=[-1]))
    
    patched_logits = run_with_head_patches(handle, corrupt_prompts, circuit_patches, return_logits_at=-1)
    patched_lD = logit_diff(patched_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
    recovery = (patched_lD - corrupt_lD) / (clean_lD - corrupt_lD)
    
    print(f"\nPatch effect recovery (circuit patched into corrupt): {recovery:.4f}")
    
    # Save results
    results = {
        "circuit": [[L, H] for L, H in circuit_heads],
        "clean_logit_diff": float(clean_lD),
        "corrupt_logit_diff": float(corrupt_lD),
        "circuit_only_logit_diff": float(circuit_lD),
        "patched_logit_diff": float(patched_lD),
        "faithfulness": float(faithfulness_value),
        "patch_effect_recovery": float(recovery),
        "metrics": {
            "faithfulness_inputs": {
                "circuit_metric": float(circuit_lD),
                "full_model_metric": float(clean_lD),
                "corrupted_metric": float(corrupt_lD)
            },
            "patch_effect_recovery_inputs": {
                "clean_metric": float(clean_lD),
                "corrupt_metric": float(corrupt_lD),
                "patched_metric": float(patched_lD)
            }
        }
    }
    
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Circuit: {circuit_heads}")
    print(f"Faithfulness: {faithfulness_value:.4f}")
    print(f"Patch effect recovery: {recovery:.4f}")
    print("\nResults saved to scratch/stage3_results.json")

if __name__ == "__main__":
    main()
