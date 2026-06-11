#!/usr/bin/env python3
"""
Stage 1.1: Logit lens analysis at END position across all layers.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.tools.lenses import logit_lens

def load_ioi_samples(split="dev", n_samples=None):
    """Load IOI dataset samples."""
    dataset_path = Path("datasets") / "ioi-abba-baba-gpt2-v1" / f"{split}.jsonl"
    
    samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    if n_samples:
        samples = samples[:n_samples]
    
    return samples

def compute_logit_lens_kl(model_handle, samples, n_samples=50):
    """
    Compute logit lens analysis at END position for a subset of samples.
    
    Returns KL divergence from final layer for each intermediate layer.
    """
    print(f"Running logit lens on {n_samples} samples...")
    
    # Sample subset for logit lens (expensive to run on all 500)
    subset = samples[:n_samples]
    
    all_kls = {layer: [] for layer in range(model_handle.n_layers)}
    io_token_ranks = {layer: [] for layer in range(model_handle.n_layers)}
    
    for i, sample in enumerate(subset):
        prompt = sample['prompt']
        io_token = sample['IO']
        
        # Get tokenizer
        tokenizer = model_handle.tokenizer
        io_token_id = tokenizer.encode(io_token, add_special_tokens=False)[0]
        
        # Run logit lens for all layers
        layers = list(range(model_handle.n_layers))
        results = logit_lens(model_handle, prompt, layers=layers, top_k=10)
        
        for result in results:
            layer = result['layer']
            kl = result['kl_to_final']
            
            # Find rank of IO token in top predictions
            top_token_ids = [t['token_id'] for t in result['top_tokens']]
            
            if io_token_id in top_token_ids:
                rank = top_token_ids.index(io_token_id) + 1
            else:
                rank = -1  # Not in top 10
            
            all_kls[layer].append(kl)
            io_token_ranks[layer].append(rank)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_samples}")
    
    # Compute mean KL and rank statistics per layer
    mean_kls = {layer: np.mean(kls) for layer, kls in all_kls.items()}
    mean_ranks = {layer: np.mean([r for r in ranks if r > 0]) 
                  for layer, ranks in io_token_ranks.items()}
    
    return mean_kls, mean_ranks, all_kls

def main():
    print("Loading GPT-2-small model...")
    model_handle = load_model("gpt2")
    print(f"Model has {model_handle.n_layers} layers, {model_handle.n_heads} heads per layer")
    
    print("\nLoading IOI dev dataset...")
    samples = load_ioi_samples("dev")
    print(f"Loaded {len(samples)} samples")
    
    # Run logit lens on subset (50 samples to keep it manageable)
    mean_kls, mean_ranks, all_kls = compute_logit_lens_kl(model_handle, samples, n_samples=50)
    
    print("\n" + "="*60)
    print("LOGIT LENS RESULTS (END position)")
    print("="*60)
    print("Layer | Mean KL to Final | Mean IO Rank (when in top-10)")
    print("-"*60)
    for layer in sorted(mean_kls.keys()):
        kl = mean_kls[layer]
        rank = mean_ranks.get(layer, -1)
        rank_str = f"{rank:.1f}" if rank > 0 else "N/A"
        print(f"{layer:5d} | {kl:16.4f} | {rank_str:>20}")
    print("="*60)
    
    # Save results
    output_path = Path("scratch") / "stage1_logit_lens.json"
    with open(output_path, 'w') as f:
        json.dump({
            'mean_kls': mean_kls,
            'mean_ranks': mean_ranks,
            'all_kls': all_kls,
            'n_samples': 50
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")
    
    # Summary
    print("\nKey observations:")
    earliest_low_kl = min((layer for layer, kl in mean_kls.items() if kl < 1.0), default=None)
    if earliest_low_kl is not None:
        print(f"  - IO information emerges around layer {earliest_low_kl} (KL < 1.0)")
    else:
        print(f"  - IO information fully emerges only in final layers")
    
    return mean_kls

if __name__ == "__main__":
    main()
