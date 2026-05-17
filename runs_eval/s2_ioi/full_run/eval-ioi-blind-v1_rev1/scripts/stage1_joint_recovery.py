#!/usr/bin/env python3
"""
Stage 1.3: Test joint recovery of top-K heads.

Patch multiple top heads together to see if their combined effect 
recovers >= 0.85 of the clean-corrupt logit_diff gap.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.tools.head_patching import cache_head_z, run_with_head_patches, HeadPatch, logit_diff

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
    """Prepare clean prompts, corrupt prompts, and target/contrast token IDs."""
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

def test_joint_recovery(model_handle, clean_prompts, corrupt_prompts, 
                        target_ids, contrast_ids, top_heads, clean_cache):
    """
    Test the joint recovery of a set of heads.
    
    Patch all heads in top_heads simultaneously from clean to corrupt.
    """
    device = model_handle.device
    
    # Create patches for all top heads at position -1
    patches = []
    for layer, head in top_heads:
        # Get the source activations from clean cache
        source = clean_cache[layer][:, -1, head, :]  # [batch, d_head]
        patches.append(HeadPatch(layer=layer, head=head, source=source, positions=[-1]))
    
    print(f"Testing joint recovery with {len(patches)} heads...")
    for layer, head in top_heads[:10]:  # Print first 10
        print(f"  L{layer}H{head}")
    if len(top_heads) > 10:
        print(f"  ... and {len(top_heads)-10} more")
    
    # Run patching
    logits = run_with_head_patches(
        model_handle,
        corrupt_prompts,
        patches
    )
    
    # Compute metric
    target_ids_dev = target_ids.to(device)
    contrast_ids_dev = contrast_ids.to(device)
    # logits is already [batch, vocab] from run_with_head_patches at the patch positions
    patched_metric = logit_diff(logits, target_ids_dev, contrast_ids_dev).mean().item()
    
    return patched_metric

def main():
    print("Loading GPT-2-small model...")
    model_handle = load_model("gpt2")
    
    print("\nLoading IOI dev dataset...")
    samples = load_ioi_samples("dev")
    print(f"Loaded {len(samples)} samples")
    
    # Prepare prompts
    print("\nPreparing prompts...")
    clean_prompts, corrupt_prompts, target_ids, contrast_ids = prepare_prompts_and_ids(
        samples, model_handle.tokenizer
    )
    
    # Cache clean activations
    print("\nCaching clean activations...")
    clean_cache = cache_head_z(model_handle, clean_prompts)
    
    # Compute baseline metrics
    print("\nComputing baseline metrics...")
    device = model_handle.device
    
    # Clean metric
    inputs_clean = model_handle.tokenizer(clean_prompts, return_tensors='pt', padding=True)
    inputs_clean = {k: v.to(device) for k, v in inputs_clean.items()}
    with torch.no_grad():
        clean_logits = model_handle.model(**inputs_clean).logits[:, -1, :]  # [batch, vocab]
    clean_metric = logit_diff(clean_logits, target_ids.to(device), contrast_ids.to(device)).mean().item()
    
    # Corrupt metric  
    inputs_corrupt = model_handle.tokenizer(corrupt_prompts, return_tensors='pt', padding=True)
    inputs_corrupt = {k: v.to(device) for k, v in inputs_corrupt.items()}
    with torch.no_grad():
        corrupt_logits = model_handle.model(**inputs_corrupt).logits[:, -1, :]  # [batch, vocab]
    corrupt_metric = logit_diff(corrupt_logits, target_ids.to(device), contrast_ids.to(device)).mean().item()
    
    gap = clean_metric - corrupt_metric
    
    print(f"\nBaseline metrics:")
    print(f"  Clean logit_diff:   {clean_metric:.4f}")
    print(f"  Corrupt logit_diff: {corrupt_metric:.4f}")
    print(f"  Gap:                {gap:.4f}")
    
    # Load recovery matrix to get top heads
    recovery_matrix = np.load("scratch/recovery_matrix.npy")
    n_layers, n_heads = recovery_matrix.shape
    
    # Get top heads by recovery
    recovery_flat = recovery_matrix.flatten()
    top_indices = np.argsort(recovery_flat)[::-1]  # Descending order
    
    # Test different numbers of top heads
    results = []
    for k in [1, 3, 5, 10, 15, 20, 30]:
        top_k_indices = top_indices[:k]
        top_heads = [(idx // n_heads, idx % n_heads) for idx in top_k_indices]
        
        patched_metric = test_joint_recovery(
            model_handle, clean_prompts, corrupt_prompts,
            target_ids, contrast_ids, top_heads, clean_cache
        )
        
        recovery = (patched_metric - corrupt_metric) / gap
        
        print(f"\nTop-{k} heads joint recovery:")
        print(f"  Patched logit_diff: {patched_metric:.4f}")
        print(f"  Recovery:           {recovery:.4f} ({recovery*100:.1f}%)")
        
        results.append({
            'k': k,
            'patched_metric': patched_metric,
            'recovery': recovery,
            'top_heads': top_heads
        })
    
    # Save results
    output_path = Path("scratch") / "stage1_joint_recovery.json"
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'clean_metric': clean_metric,
            'corrupt_metric': corrupt_metric,
            'gap': gap
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Find minimum K that achieves >= 0.85 recovery
    for r in results:
        if r['recovery'] >= 0.85:
            print(f"\n✓ Top-{r['k']} heads achieve {r['recovery']:.3f} recovery (>= 0.85 threshold)")
            return r
    
    best = max(results, key=lambda x: x['recovery'])
    print(f"\n✗ Best recovery: {best['recovery']:.3f} with top-{best['k']} heads")
    print("  Does not meet 0.85 threshold")
    
    return None

if __name__ == "__main__":
    main()
