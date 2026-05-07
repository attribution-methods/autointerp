"""Stage 4: Validation on heldout split.

Evaluate circuit faithfulness and minimality on 100 heldout samples.
Using END-ONLY ablation as specified in the stage notes.
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
    
    # Generate HELDOUT pairs (100 samples, disjoint names from dev)
    print("\nGenerating 100 IOI HELDOUT pairs...")
    pairs = generate_pairs(None, n_pairs=100, seed=999, tokenizer=handle.tokenizer, split="heldout")
    
    clean_prompts = [p.clean_prompt for p in pairs]
    corrupt_prompts = [p.corrupted_prompt for p in pairs]
    io_ids = torch.tensor([p.target_token_id for p in pairs])
    s_ids = torch.tensor([p.foil_token_id for p in pairs])
    
    print(f"Generated {len(pairs)} heldout pairs")
    print(f"Example clean: {clean_prompts[0]!r}")
    
    # Load circuit from stage1
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage1_results.json") as f:
        stage1 = json.load(f)
    
    circuit_heads = [(L, H) for L, H in stage1["top_K_heads"]]
    print(f"\nCircuit C = {circuit_heads}")
    
    # Compute baselines on heldout
    print("\n=== Computing baselines on heldout ===")
    
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
    
    # === FAITHFULNESS: Circuit-only performance (complement ablated at END) ===
    print("\n=== Computing circuit faithfulness (END-ONLY ablation) ===")
    
    # 1. Cache mean head z over heldout clean batch
    clean_cache = cache_head_z(handle, clean_prompts)
    mean_z = mean_head_z(clean_cache)  # {layer: [seq, n_heads, d_head]}
    
    # 2. Build patches for all heads NOT in C, at END position ONLY
    circuit_set = set(circuit_heads)
    patches_complement = []
    
    for L in range(handle.n_layers):
        for H in range(handle.n_heads):
            if (L, H) not in circuit_set:
                source_z = mean_z[L][:, H, :]  # [seq, d_head]
                patches_complement.append(HeadPatch(layer=L, head=H, source=source_z, positions=[-1]))
    
    print(f"Ablating {len(patches_complement)} heads (complement of C) at END only...")
    
    # 3. Run circuit-only forward
    circuit_logits = run_with_head_patches(handle, clean_prompts, patches_complement, return_logits_at=-1)
    circuit_lD = logit_diff(circuit_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
    
    # 4. Compute faithfulness
    faithfulness_full = (circuit_lD - corrupt_lD) / (clean_lD - corrupt_lD)
    
    print(f"Circuit-only logit_diff: {circuit_lD:.4f}")
    print(f"Faithfulness (full circuit): {faithfulness_full:.4f}")
    
    # === MINIMALITY: Remove each component individually ===
    print("\n=== Computing minimality (removing each component) ===")
    
    removed_faithfulness_list = []
    
    for i, (remove_L, remove_H) in enumerate(circuit_heads):
        print(f"\n{i+1}. Removing ({remove_L}, {remove_H}) from circuit...")
        
        # Build patches: ablate complement + ablate the removed component
        patches_minus_one = []
        for L in range(handle.n_layers):
            for H in range(handle.n_heads):
                if (L, H) not in circuit_set or (L, H) == (remove_L, remove_H):
                    source_z = mean_z[L][:, H, :]
                    patches_minus_one.append(HeadPatch(layer=L, head=H, source=source_z, positions=[-1]))
        
        # Run with component removed
        logits_minus_one = run_with_head_patches(handle, clean_prompts, patches_minus_one, return_logits_at=-1)
        lD_minus_one = logit_diff(logits_minus_one, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
        
        faithfulness_minus_one = (lD_minus_one - corrupt_lD) / (clean_lD - corrupt_lD)
        drop = faithfulness_full - faithfulness_minus_one
        
        print(f"   Faithfulness (minus this component): {faithfulness_minus_one:.4f}")
        print(f"   Drop: {drop:.4f}")
        
        removed_faithfulness_list.append(float(faithfulness_minus_one))
    
    # Minimality = minimum drop across all components
    minimality_value = min(faithfulness_full - f for f in removed_faithfulness_list)
    worst_component_idx = removed_faithfulness_list.index(max(removed_faithfulness_list))
    
    print(f"\n=== Summary ===")
    print(f"Circuit: {circuit_heads}")
    print(f"Faithfulness: {faithfulness_full:.4f}")
    print(f"Minimality: {minimality_value:.4f}")
    print(f"Weakest component: {circuit_heads[worst_component_idx]} (drop={faithfulness_full - removed_faithfulness_list[worst_component_idx]:.4f})")
    
    # Save results
    results = {
        "circuit": [[L, H] for L, H in circuit_heads],
        "heldout_n_samples": len(pairs),
        "clean_logit_diff": float(clean_lD),
        "corrupt_logit_diff": float(corrupt_lD),
        "circuit_only_logit_diff": float(circuit_lD),
        "faithfulness": float(faithfulness_full),
        "minimality": float(minimality_value),
        "removed_faithfulness": removed_faithfulness_list,
        "metrics": {
            "faithfulness_inputs": {
                "circuit_metric": float(circuit_lD),
                "full_model_metric": float(clean_lD),
                "corrupted_metric": float(corrupt_lD)
            },
            "minimality_inputs": {
                "full_circuit_faithfulness": float(faithfulness_full),
                "removed_faithfulness": removed_faithfulness_list
            }
        }
    }
    
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to scratch/stage4_results.json")
    
    # Check criteria
    print(f"\n=== Criterion Checks ===")
    print(f"circuit-faithfulness: {faithfulness_full:.4f} >= 0.85? {faithfulness_full >= 0.85}")
    print(f"circuit-minimality: {minimality_value:.4f} >= 0.05? {minimality_value >= 0.05}")

if __name__ == "__main__":
    main()
