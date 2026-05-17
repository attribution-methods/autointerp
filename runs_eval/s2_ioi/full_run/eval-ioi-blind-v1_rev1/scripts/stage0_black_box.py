#!/usr/bin/env python3
"""
Stage 0: Black-box behavioral measurement on IOI task.
Measure accuracy and logit_diff for GPT-2-small on dev split.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autointerp.tools.model import load_model

def load_ioi_dataset(split="dev"):
    """Load IOI dataset from the expected location."""
    dataset_path = Path("datasets") / "ioi-abba-baba-gpt2-v1" / f"{split}.jsonl"
    
    samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    return samples

def compute_ioi_metrics(model_handle, samples):
    """
    Compute accuracy and logit_diff for IOI task.
    
    Returns:
        accuracy: fraction of samples where logit(IO) > logit(S)
        logit_diff: mean of [logit(IO) - logit(S)]
        per_sample_results: list of dicts with detailed results
    """
    model = model_handle.model
    tokenizer = model_handle.tokenizer
    device = model_handle.device
    
    results = []
    
    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        io_token = sample['IO']  # The correct answer (indirect object)
        s_token = sample['S']    # The incorrect answer (subject)
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Get logits at the final position
        final_logits = logits[0, -1, :]  # [vocab_size]
        
        # Get token IDs for IO and S
        # These should be single-token names with leading space
        io_token_id = tokenizer.encode(io_token, add_special_tokens=False)[0]
        s_token_id = tokenizer.encode(s_token, add_special_tokens=False)[0]
        
        # Extract logits
        logit_io = final_logits[io_token_id].item()
        logit_s = final_logits[s_token_id].item()
        logit_diff = logit_io - logit_s
        
        correct = logit_diff > 0
        
        results.append({
            'sample_idx': i,
            'prompt': prompt,
            'io_token': io_token,
            's_token': s_token,
            'logit_io': logit_io,
            'logit_s': logit_s,
            'logit_diff': logit_diff,
            'correct': correct
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(samples)} samples")
    
    # Compute aggregate metrics
    accuracy = np.mean([r['correct'] for r in results])
    mean_logit_diff = np.mean([r['logit_diff'] for r in results])
    
    return accuracy, mean_logit_diff, results

def main():
    print("Loading GPT-2-small model...")
    model_handle = load_model("gpt2")
    
    print("Loading IOI dev dataset...")
    samples = load_ioi_dataset("dev")
    print(f"Loaded {len(samples)} samples")
    
    print("\nComputing IOI metrics...")
    accuracy, logit_diff, per_sample_results = compute_ioi_metrics(model_handle, samples)
    
    print("\n" + "="*60)
    print("STAGE 0: BLACK-BOX RESULTS")
    print("="*60)
    print(f"Accuracy (IO > S):     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Mean logit_diff:       {logit_diff:.4f}")
    print(f"Total samples:         {len(samples)}")
    print(f"Correct predictions:   {sum(r['correct'] for r in per_sample_results)}")
    print("="*60)
    
    # Save detailed results
    output_path = Path("scratch") / "stage0_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'mean_logit_diff': logit_diff,
            'n_samples': len(samples),
            'n_correct': sum(r['correct'] for r in per_sample_results),
            'per_sample_results': per_sample_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")
    
    # Check abort conditions
    print("\n" + "="*60)
    print("ABORT CONDITION CHECKS")
    print("="*60)
    print(f"Accuracy >= 0.9:       {'PASS' if accuracy >= 0.9 else 'FAIL'}")
    print(f"Logit_diff >= 2.0:     {'PASS' if logit_diff >= 2.0 else 'FAIL'}")
    print("="*60)
    
    return accuracy, logit_diff

if __name__ == "__main__":
    main()
