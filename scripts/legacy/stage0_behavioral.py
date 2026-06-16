"""
Stage 0: Black-box behavioral measurement on IOI task.
Measures accuracy and logit_diff on dev split.
"""
import sys

sys.path.insert(0, "src")

import json

import numpy as np
import torch

from autointerp.tools.model import load_model


def main():
    print("Loading GPT-2-small...")
    handle = load_model("gpt2")
    
    print("Loading IOI dataset (dev split, n=500)...")
    with open("scratch/ioi_dev_500.json") as f:
        dev_samples = json.load(f)
    
    print(f"Dataset loaded: {len(dev_samples)} samples")
    print(f"Example sample: {dev_samples[0]}")
    
    # Extract prompts and targets
    results = []
    accuracy_count = 0
    logit_diffs = []
    
    print("\nProcessing samples...")
    for i, sample in enumerate(dev_samples):
        if i % 50 == 0:
            print(f"  {i}/{len(dev_samples)}...")
        
        # Get the prompt text
        prompt = sample["prompt"]
        io_name = sample["IO"]  # Indirect Object name
        s_name = sample["S"]    # Subject name
        
        # Tokenize
        tokens = handle.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(handle.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = handle.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Final position logits
        
        # Get logits for IO and S tokens
        # Tokenize IO and S names to get their token IDs
        io_token_id = handle.tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
        s_token_id = handle.tokenizer.encode(" " + s_name, add_special_tokens=False)[0]
        
        io_logit = logits[io_token_id].item()
        s_logit = logits[s_token_id].item()
        
        logit_diff = io_logit - s_logit
        logit_diffs.append(logit_diff)
        
        if logit_diff > 0:
            accuracy_count += 1
        
        results.append({
            "sample_idx": i,
            "prompt": prompt,
            "io_name": io_name,
            "s_name": s_name,
            "io_logit": io_logit,
            "s_logit": s_logit,
            "logit_diff": logit_diff,
            "correct": logit_diff > 0
        })
    
    # Compute metrics
    accuracy = accuracy_count / len(dev_samples)
    mean_logit_diff = np.mean(logit_diffs)
    std_logit_diff = np.std(logit_diffs)
    
    print(f"\n{'='*60}")
    print("STAGE 0 BEHAVIORAL RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(dev_samples)}")
    print(f"Accuracy (logit(IO) > logit(S)): {accuracy:.4f} ({accuracy_count}/{len(dev_samples)})")
    print(f"Mean logit_diff: {mean_logit_diff:.4f} ± {std_logit_diff:.4f}")
    print(f"Min logit_diff: {min(logit_diffs):.4f}")
    print(f"Max logit_diff: {max(logit_diffs):.4f}")
    print(f"{'='*60}")
    
    # Save detailed results
    output = {
        "n_samples": len(dev_samples),
        "accuracy": accuracy,
        "accuracy_count": accuracy_count,
        "mean_logit_diff": mean_logit_diff,
        "std_logit_diff": std_logit_diff,
        "min_logit_diff": min(logit_diffs),
        "max_logit_diff": max(logit_diffs),
        "results": results
    }
    
    with open("scratch/stage0_behavioral_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to scratch/stage0_behavioral_results.json")
    
    # Print summary for metric computation
    print("\n[METRIC_SUMMARY]")
    print(json.dumps({
        "accuracy": accuracy,
        "mean_logit_diff": mean_logit_diff
    }))

if __name__ == "__main__":
    main()
