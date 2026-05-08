#!/usr/bin/env python3
"""
Stage 0: Black-box IOI evaluation on GPT-2-small.
Compute accuracy and logit_diff for behavioral-sanity criterion.
"""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import json
from pathlib import Path

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs

def main():
    print("=" * 80)
    print("STAGE 0: BLACK-BOX IOI EVALUATION")
    print("=" * 80)
    
    # Load spec
    spec_path = Path("runs/ioi-discovery-v1_rev3/spec.json")
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load model
    print("\n[1/5] Loading GPT-2-small...")
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded: {handle.model_id}, device={handle.device}, dtype={handle.dtype}")
    
    # Load IOI dataset
    print("\n[2/5] Loading IOI dev dataset (n=500, seed=42)...")
    pairs = generate_pairs(
        spec,
        n_pairs=500,
        split="dev",
        seed=42,
        tokenizer=handle.tokenizer
    )
    
    print(f"Loaded {len(pairs)} clean/corrupt pairs")
    print(f"Example clean: {pairs[0].clean_prompt}")
    print(f"Example corrupt: {pairs[0].corrupted_prompt}")
    print(f"Example IO name: {pairs[0].metadata['io_name']} (id={pairs[0].target_token_id})")
    print(f"Example S name: {pairs[0].metadata['s_name']} (id={pairs[0].foil_token_id})")
    
    # Extract prompts
    clean_prompts = [p.clean_prompt for p in pairs]
    IO_token_ids = [p.target_token_id for p in pairs]
    S_token_ids = [p.foil_token_id for p in pairs]
    
    # Run clean forward passes
    print("\n[3/5] Running clean forward passes...")
    clean_logits_list = []
    batch_size = 32
    
    for i in range(0, len(clean_prompts), batch_size):
        batch = clean_prompts[i:i+batch_size]
        inputs = handle.tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(handle.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = handle.model(**inputs)
            logits = outputs.logits  # shape: (batch, seq_len, vocab)
            
            # Extract last-position logits for each sequence
            last_positions = inputs['attention_mask'].sum(dim=1) - 1  # shape: (batch,)
            batch_logits = torch.stack([
                logits[j, last_positions[j], :]
                for j in range(len(batch))
            ])  # shape: (batch, vocab)
            
            clean_logits_list.append(batch_logits.cpu())
    
    clean_logits = torch.cat(clean_logits_list, dim=0)  # shape: (500, vocab)
    print(f"Clean logits shape: {clean_logits.shape}")
    
    # Extract IO and S logits (SCALARS per example)
    print("\n[4/5] Extracting IO vs S logits...")
    IO_logits = torch.tensor([
        clean_logits[i, IO_token_ids[i]].item()
        for i in range(len(clean_prompts))
    ])
    S_logits = torch.tensor([
        clean_logits[i, S_token_ids[i]].item()
        for i in range(len(clean_prompts))
    ])
    
    print(f"IO logits shape: {IO_logits.shape}, mean={IO_logits.mean():.3f}")
    print(f"S logits shape: {S_logits.shape}, mean={S_logits.mean():.3f}")
    
    # Compute metrics
    print("\n[5/5] Computing metrics...")
    
    # Accuracy: fraction where IO logit > S logit
    correct = (IO_logits > S_logits).float()
    accuracy = correct.mean().item()
    
    # Logit diff: mean(IO_logit - S_logit)
    logit_diff = (IO_logits - S_logits).mean().item()
    
    print(f"\n{'='*80}")
    print(f"RESULTS (dev split, n={len(clean_prompts)})")
    print(f"{'='*80}")
    print(f"Accuracy (IO > S):     {accuracy:.4f}  ({correct.sum().int()}/{len(clean_prompts)})")
    print(f"Logit diff (IO - S):   {logit_diff:.4f}")
    print(f"{'='*80}")
    
    # Check abort condition
    if accuracy < 0.9:
        print(f"\n⚠️  WARNING: Accuracy {accuracy:.4f} < 0.9 (abort threshold)")
    else:
        print(f"\n✓ Accuracy {accuracy:.4f} >= 0.9 (passes abort threshold)")
    
    # Save results
    results = {
        "accuracy": accuracy,
        "logit_diff": logit_diff,
        "n_samples": len(clean_prompts),
        "IO_logits_mean": IO_logits.mean().item(),
        "S_logits_mean": S_logits.mean().item(),
    }
    
    output_path = Path("runs/ioi-discovery-v1_rev3/scratch/stage0_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nNext: Commit accuracy and logit_diff MetricResults, evaluate behavioral-sanity criterion.")

if __name__ == "__main__":
    main()
