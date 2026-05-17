#!/usr/bin/env python3
"""
Stage 1.2: Activation patching from corrupt to clean for all (layer, head) pairs.

Sweep ALL heads across all 12 layers to find which attention heads causally contribute
to the IOI behavior (preferring IO over S).
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.tools.head_patching import cache_head_z, head_patch_sweep, logit_diff

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

def prepare_prompts_and_ids(samples, tokenizer):
    """
    Prepare clean prompts, corrupt prompts, and target/contrast token IDs.
    """
    clean_prompts = [s['prompt'] for s in samples]
    corrupt_prompts = [s['corrupt_prompt'] for s in samples]
    
    target_ids = []
    contrast_ids = []
    
    for s in samples:
        io_token_id = tokenizer.encode(s['IO'], add_special_tokens=False)[0]
        s_token_id = tokenizer.encode(s['S'], add_special_tokens=False)[0]
        target_ids.append(io_token_id)
        contrast_ids.append(s_token_id)
    
    return clean_prompts, corrupt_prompts, torch.tensor(target_ids), torch.tensor(contrast_ids)

def main():
    print("Loading GPT-2-small model...")
    model_handle = load_model("gpt2")
    n_layers = model_handle.n_layers
    n_heads = model_handle.n_heads
    print(f"Model: {n_layers} layers, {n_heads} heads per layer")
    print(f"Total heads to sweep: {n_layers * n_heads}")
    
    print("\nLoading IOI dev dataset...")
    samples = load_ioi_samples("dev")
    print(f"Loaded {len(samples)} samples")
    
    # Prepare prompts and token IDs
    print("\nPreparing prompts...")
    clean_prompts, corrupt_prompts, target_ids, contrast_ids = prepare_prompts_and_ids(
        samples, model_handle.tokenizer
    )
    
    # Cache activations on clean and corrupt
    print("\nCaching clean activations...")
    clean_cache = cache_head_z(model_handle, clean_prompts)
    print("Caching corrupt activations...")
    corrupt_cache = cache_head_z(model_handle, corrupt_prompts)
    
    # Define metric: logit_diff at final position
    def metric_fn(logits):
        return logit_diff(logits, target_ids, contrast_ids)
    
    # Run head patching sweep
    print("\nRunning head patch sweep...")
    print("This will patch clean activations INTO corrupt runs for each (layer, head)")
    print("and measure recovery of logit_diff...")
    
    sweep_result = head_patch_sweep(
        model_handle,
        clean_prompts,
        corrupt_prompts,
        metric_fn,
        patch_positions=[-1],  # Patch at END position
        clean_cache=clean_cache
    )
    
    recovery = sweep_result['recovery']  # [n_layers, n_heads] tensor
    
    print("\n" + "="*80)
    print("HEAD PATCH SWEEP RESULTS")
    print("="*80)
    print("Recovery matrix (higher = more important for IOI behavior):")
    print(f"Shape: {recovery.shape} (layers x heads)")
    print()
    
    # Print as matrix
    print("     ", end="")
    for h in range(n_heads):
        print(f"   H{h:2d}", end="")
    print()
    print("-" * 80)
    
    for layer in range(n_layers):
        print(f"L{layer:2d}  ", end="")
        for head in range(n_heads):
            val = recovery[layer, head].item()
            print(f" {val:5.3f}", end="")
        print()
    
    print("="*80)
    
    # Identify top heads
    recovery_flat = recovery.view(-1)
    top_k = 20
    top_indices = torch.topk(recovery_flat, k=top_k).indices
    
    print(f"\nTop {top_k} heads by recovery:")
    print("Rank | Layer | Head | Recovery")
    print("-" * 40)
    for rank, idx in enumerate(top_indices):
        layer = idx.item() // n_heads
        head = idx.item() % n_heads
        rec = recovery[layer, head].item()
        print(f"{rank+1:4d} | {layer:5d} | {head:4d} | {rec:8.4f}")
    
    # Compute statistics
    mean_recovery = recovery.mean().item()
    max_recovery = recovery.max().item()
    top10_mean = recovery_flat[torch.topk(recovery_flat, k=10).indices].mean().item()
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Mean recovery across all heads:  {mean_recovery:.4f}")
    print(f"Max recovery (single head):      {max_recovery:.4f}")
    print(f"Top-10 heads mean recovery:      {top10_mean:.4f}")
    print("="*80)
    
    # Save results
    clean_metric = float(sweep_result['clean_metric'])
    corrupt_metric = float(sweep_result['corrupt_metric'])
    gap = clean_metric - corrupt_metric
    
    output_path = Path("scratch") / "stage1_head_patching.json"
    with open(output_path, 'w') as f:
        json.dump({
            'recovery_matrix': recovery.tolist(),
            'mean_recovery': mean_recovery,
            'max_recovery': max_recovery,
            'top10_mean_recovery': top10_mean,
            'clean_logit_diff': clean_metric,
            'corrupt_logit_diff': corrupt_metric,
            'gap': gap,
            'n_samples': len(samples),
            'n_layers': n_layers,
            'n_heads': n_heads
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")
    
    # Save recovery matrix for later use
    np.save("scratch/recovery_matrix.npy", recovery.cpu().numpy())
    print("Recovery matrix saved to scratch/recovery_matrix.npy")
    
    return recovery, sweep_result

if __name__ == "__main__":
    main()
