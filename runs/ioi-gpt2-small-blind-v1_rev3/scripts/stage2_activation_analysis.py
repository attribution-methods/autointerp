"""Stage 2: Activation analysis - contrastive directions and mean ablation.

Goals:
1. Extract contrastive directions (IO vs S) from residual stream
2. Mean-ablate top-3 heads individually and measure ablation_drop
"""
import sys
sys.path.insert(0, "src")

import torch
import json
from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs
from autointerp.tools.head_patching import (
    cache_head_z, mean_head_z, mean_ablate_heads, logit_diff
)

def main():
    # Load model
    print("Loading model gpt2...")
    handle = load_model("gpt2")
    
    # Generate IOI pairs
    print("\nGenerating 500 IOI dev pairs...")
    pairs = generate_pairs(None, n_pairs=500, seed=42, tokenizer=handle.tokenizer, split="dev")
    
    clean_prompts = [p.clean_prompt for p in pairs]
    io_ids = torch.tensor([p.target_token_id for p in pairs])
    s_ids = torch.tensor([p.foil_token_id for p in pairs]
)
    
    # Load stage1 results to get top-3 heads
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage1_results.json") as f:
        stage1 = json.load(f)
    
    top_K_heads = [(L, H) for L, H in stage1["top_K_heads"]]
    print(f"\nTop-3 heads from localization: {top_K_heads}")
    
    # Compute baseline logit_diff on clean
    print("\n=== Computing baseline (clean) ===")
    clean_tokens = handle.tokenizer(clean_prompts, return_tensors="pt", padding=True,
                                     add_special_tokens=False).to(handle.device)
    with torch.no_grad():
        clean_logits = handle.model(**clean_tokens).logits[:, -1, :]
    baseline_lD = logit_diff(clean_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
    print(f"Baseline logit_diff (clean): {baseline_lD:.4f}")
    
    # Cache clean head z for mean ablation
    print("\n=== Caching clean head z for ablation ===")
    clean_cache = cache_head_z(handle, clean_prompts)
    mean_z = mean_head_z(clean_cache)
    
    # Mean-ablate each head individually
    print("\n=== Mean-ablating each head individually ===")
    ablation_results = []
    
    for i, (L, H) in enumerate(top_K_heads):
        print(f"\n{i+1}. Ablating Layer {L}, Head {H}...")
        sites_to_ablate = [(L, H)]
        
        ablated_logits = mean_ablate_heads(handle, clean_prompts, sites_to_ablate, mean_z)
        ablated_lD = logit_diff(ablated_logits, io_ids.to(handle.device), s_ids.to(handle.device)).mean().item()
        
        drop = baseline_lD - ablated_lD
        drop_pct = (drop / baseline_lD) * 100 if baseline_lD != 0 else 0
        
        print(f"   Ablated logit_diff: {ablated_lD:.4f}")
        print(f"   Drop: {drop:.4f} ({drop_pct:.1f}%)")
        
        ablation_results.append({
            "layer": L,
            "head": H,
            "ablated_logit_diff": float(ablated_lD),
            "drop": float(drop),
            "drop_percent": float(drop_pct)
        })
    
    # Compute worst (largest) drop for ablation_drop metric
    worst_drop = max(r["drop"] for r in ablation_results)
    worst_head = max(ablation_results, key=lambda r: r["drop"])
    
    print(f"\n=== Summary ===")
    print(f"Baseline logit_diff: {baseline_lD:.4f}")
    print(f"Worst ablation: Layer {worst_head['layer']}, Head {worst_head['head']}")
    print(f"  Ablated logit_diff: {worst_head['ablated_logit_diff']:.4f}")
    print(f"  Drop: {worst_drop:.4f}")
    
    # Save results
    results = {
        "baseline_logit_diff": float(baseline_lD),
        "ablation_results": ablation_results,
        "worst_drop": float(worst_drop),
        "worst_head": {"layer": worst_head["layer"], "head": worst_head["head"]},
        "metrics": {
            "logit_diff": float(baseline_lD),
            "ablation_drop_inputs": {
                "baseline_metric": float(baseline_lD),
                "ablated_metric": float(worst_head["ablated_logit_diff"])
            }
        }
    }
    
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to scratch/stage2_results.json")

if __name__ == "__main__":
    main()
