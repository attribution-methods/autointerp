"""
Stage 1: Localization via head patching.
Create ABC-corrupted prompts and run head_patch_sweep.
"""
import sys
sys.path.insert(0, '../../src')

import torch
import json
import numpy as np
from autointerp.tools.model import load_model
from autointerp.tools.head_patching import (
    cache_head_z, head_patch_sweep, run_with_head_patches, 
    HeadPatch, logit_diff
)

def load_ioi_dataset(split="dev", n_samples=500):
    """Load IOI dataset - same as stage 0."""
    names = [' John', ' Mary', ' Tom', ' Sarah', ' James', ' Kate', ' Robert', ' Lisa',
             ' Michael', ' Emily', ' David', ' Anna', ' Chris', ' Laura', ' Mark', ' Emma']
    places = [' park', ' store', ' mall', ' school', ' beach', ' museum', ' office', ' library']
    objects = [' book', ' pen', ' gift', ' letter', ' toy', ' drink', ' card', ' bag']
    
    examples = []
    np.random.seed(42 if split == 'dev' else 43)
    
    for i in range(n_samples):
        name_indices = np.random.choice(len(names), 2, replace=False)
        A, B = names[name_indices[0]], names[name_indices[1]]
        place = np.random.choice(places)
        obj = np.random.choice(objects)
        
        if i % 2 == 0:
            prompt = f'When{A} and{B} went to the{place},{B} gave a{obj} to'
            io_name, s_name, template = A, B, 'BABA'
        else:
            prompt = f'When{A} and{B} went to the{place},{A} gave a{obj} to'
            io_name, s_name, template = B, A, 'ABBA'
        
        examples.append({
            'prompt': prompt, 'io_name': io_name, 's_name': s_name, 
            'template': template, 'A': A, 'B': B, 'place': place, 'object': obj
        })
    return examples

def create_abc_corruption(examples, tokenizer):
    """Create ABC-corrupted prompts where one repeated name is replaced with C."""
    # Get available names for C (must be different from A and B)
    all_names = [' John', ' Mary', ' Tom', ' Sarah', ' James', ' Kate', ' Robert', ' Lisa',
                 ' Michael', ' Emily', ' David', ' Anna', ' Chris', ' Laura', ' Mark', ' Emma']
    
    corrupted = []
    for ex in examples:
        # Find a name C that's not A or B
        available = [n for n in all_names if n not in [ex['A'], ex['B']]]
        C = np.random.choice(available)
        
        # ABC corruption: replace one occurrence of the repeated name with C
        # For BABA template: replace first occurrence of B with C
        # For ABBA template: replace first occurrence of A with C
        if ex['template'] == 'BABA':
            # Original: When A and B went to..., B gave...
            # Corrupt: When A and C went to..., B gave...
            corrupt_prompt = f"When{ex['A']} and{C} went to the{ex['place']},{ex['B']} gave a{ex['object']} to"
        else:  # ABBA
            # Original: When A and B went to..., A gave...
            # Corrupt: When C and B went to..., A gave...
            corrupt_prompt = f"When{C} and{ex['B']} went to the{ex['place']},{ex['A']} gave a{ex['object']} to"
        
        # Verify same tokenization length
        clean_toks = tokenizer.encode(ex['prompt'], add_special_tokens=False)
        corrupt_toks = tokenizer.encode(corrupt_prompt, add_special_tokens=False)
        assert len(clean_toks) == len(corrupt_toks), f"Length mismatch: {len(clean_toks)} vs {len(corrupt_toks)}"
        
        corrupted.append({
            'prompt': corrupt_prompt,
            'io_name': ex['io_name'],
            's_name': ex['s_name'],
            'C': C
        })
    
    return corrupted

def main():
    print("Loading GPT-2-small...")
    handle = load_model('gpt2')
    
    print("Loading IOI dev dataset...")
    clean_examples = load_ioi_dataset(split='dev', n_samples=500)
    
    print("Creating ABC-corrupted dataset...")
    corrupt_examples = create_abc_corruption(clean_examples, handle.tokenizer)
    
    print(f"Sample clean: {clean_examples[0]['prompt']}")
    print(f"Sample corrupt: {corrupt_examples[0]['prompt']}")
    
    # Extract prompts
    clean_prompts = [ex['prompt'] for ex in clean_examples]
    corrupt_prompts = [ex['prompt'] for ex in corrupt_examples]
    
    # Get IO and S token IDs for computing logit_diff
    io_token_ids = [handle.tokenizer.encode(ex['io_name'], add_special_tokens=False)[0] 
                    for ex in clean_examples]
    s_token_ids = [handle.tokenizer.encode(ex['s_name'], add_special_tokens=False)[0] 
                   for ex in clean_examples]
    io_token_ids = torch.tensor(io_token_ids, device=handle.device)
    s_token_ids = torch.tensor(s_token_ids, device=handle.device)
    
    print("\\nCaching clean and corrupt head activations...")
    clean_cache = cache_head_z(handle, clean_prompts)
    corrupt_cache = cache_head_z(handle, corrupt_prompts)
    
    print(f"Cached {len(clean_cache)} layers")
    print(f"Sample cache shape: {clean_cache[0].shape}")  # [B, S, H, d_head]
    
    # Define metric function for logit_diff
    def metric_fn(logits):
        """Compute logit_diff for each example. logits shape: [B, V]"""
        target_logits = logits[torch.arange(len(logits), device=logits.device), io_token_ids]
        foil_logits = logits[torch.arange(len(logits), device=logits.device), s_token_ids]
        return target_logits - foil_logits
    
    print("\\nRunning head patch sweep (144 heads)...")
    sweep_result = head_patch_sweep(
        handle, clean_prompts, corrupt_prompts, metric_fn,
        patch_positions=[-1],  # Patch at final position
        clean_cache=clean_cache
    )
    
    recovery_matrix = sweep_result['recovery']  # [n_layers, n_heads]
    print(f"Recovery matrix shape: {recovery_matrix.shape}")
    print(f"\\nRecovery statistics:")
    print(f"  Max: {recovery_matrix.max():.4f}")
    print(f"  Mean: {recovery_matrix.mean():.4f}")
    print(f"  Min: {recovery_matrix.min():.4f}")
    
    # Find top-K heads
    K = 5  # Take top-5 for analysis
    flat_recovery = recovery_matrix.flatten()
    topk_indices = torch.topk(flat_recovery, K).indices
    n_heads = recovery_matrix.shape[1]
    top_heads = [(idx.item() // n_heads, idx.item() % n_heads, flat_recovery[idx].item()) 
                 for idx in topk_indices]
    
    print(f"\\nTop-{K} heads by single-head recovery:")
    for i, (layer, head, recovery) in enumerate(top_heads):
        print(f"  {i+1}. L{layer}H{head}: {recovery:.4f}")
    
    # Save head sweep results
    clean_metric = sweep_result['clean_metric']
    corrupt_metric = sweep_result['corrupt_metric']
    if isinstance(clean_metric, torch.Tensor):
        clean_metric = clean_metric.mean().item()
    if isinstance(corrupt_metric, torch.Tensor):
        corrupt_metric = corrupt_metric.mean().item()
    
    results = {
        'recovery_matrix': recovery_matrix.cpu().tolist(),
        'top_k_heads': [(l, h, float(r)) for l, h, r in top_heads],
        'clean_metric': float(clean_metric),
        'corrupt_metric': float(corrupt_metric)
    }
    
    with open('scratch/stage1_head_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nSaved head sweep results to scratch/stage1_head_sweep.json")
    
    # Now compute JOINT recovery with top-3 heads
    print("\\n" + "="*60)
    print("Computing JOINT recovery with top-3 heads...")
    print("="*60)
    
    top_3 = top_heads[:3]
    print(f"Top-3 heads: {[(l,h) for l,h,_ in top_3]}")
    
    # Cache clean z for the top-3 layers
    top_layers = list(set([l for l, h, _ in top_3]))
    clean_z = cache_head_z(handle, clean_prompts, layers=top_layers)
    
    # Build patches for all 3 heads
    patches = [HeadPatch(layer=L, head=H, source=clean_z[L][:,:,H,:]) 
               for L, H, _ in top_3]
    
    # Run with joint patches
    patched_logits = run_with_head_patches(
        handle, corrupt_prompts, patches, return_logits_at=-1
    )
    
    # Compute logit_diff on patched outputs
    joint_lD = logit_diff(patched_logits, io_token_ids, s_token_ids).mean().item()
    
    # Get clean and corrupt baselines
    clean_lD = results['clean_metric']
    corrupt_lD = results['corrupt_metric']
    
    # Compute recovery
    joint_recovery = (joint_lD - corrupt_lD) / (clean_lD - corrupt_lD)
    
    print(f"\\nJoint recovery results:")
    print(f"  Clean logit_diff: {clean_lD:.4f}")
    print(f"  Corrupt logit_diff: {corrupt_lD:.4f}")
    print(f"  Patched logit_diff (joint top-3): {joint_lD:.4f}")
    print(f"  Joint recovery: {joint_recovery:.4f}")
    
    # Save joint recovery results
    joint_results = {
        'top_3_heads': [(l, h) for l, h, _ in top_3],
        'clean_logit_diff': clean_lD,
        'corrupt_logit_diff': corrupt_lD,
        'patched_logit_diff': joint_lD,
        'joint_recovery': joint_recovery
    }
    
    with open('scratch/stage1_joint_recovery.json', 'w') as f:
        json.dump(joint_results, f, indent=2)
    
    print(f"\\nSaved joint recovery to scratch/stage1_joint_recovery.json")
    
    return results, joint_results

if __name__ == "__main__":
    main()
