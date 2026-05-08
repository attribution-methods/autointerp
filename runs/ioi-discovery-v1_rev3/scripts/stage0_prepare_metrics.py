#!/usr/bin/env python3
"""Prepare per-sample metric inputs for compute_metric calls."""

import sys
sys.path.insert(0, "src")

import torch
import json
from pathlib import Path

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs

def main():
    # Load spec
    spec_path = Path("runs/ioi-discovery-v1_rev3/spec.json")
    with open(spec_path) as f:
        spec = json.load(f)
    
    # Load model
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load IOI dataset
    pairs = generate_pairs(
        spec,
        n_pairs=500,
        split="dev",
        seed=42,
        tokenizer=handle.tokenizer
    )
    
    clean_prompts = [p.clean_prompt for p in pairs]
    IO_token_ids = [p.target_token_id for p in pairs]
    S_token_ids = [p.foil_token_id for p in pairs]
    
    # Run forward passes to get per-sample logits
    print("Running forward passes...")
    clean_logits_list = []
    batch_size = 32
    
    for i in range(0, len(clean_prompts), batch_size):
        batch = clean_prompts[i:i+batch_size]
        inputs = handle.tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(handle.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = handle.model(**inputs)
            logits = outputs.logits
            last_positions = inputs['attention_mask'].sum(dim=1) - 1
            batch_logits = torch.stack([
                logits[j, last_positions[j], :]
                for j in range(len(batch))
            ])
            clean_logits_list.append(batch_logits.cpu())
    
    clean_logits = torch.cat(clean_logits_list, dim=0)
    
    # Extract per-sample IO and S logits
    IO_logits = [clean_logits[i, IO_token_ids[i]].item() for i in range(len(clean_prompts))]
    S_logits = [clean_logits[i, S_token_ids[i]].item() for i in range(len(clean_prompts))]
    
    # For accuracy: predictions (0/1 per sample) and labels (all 1s for clean, IO should win)
    predictions = [1.0 if IO_logits[i] > S_logits[i] else 0.0 for i in range(len(clean_prompts))]
    labels = [1.0] * len(clean_prompts)  # All samples should have IO > S
    
    # Save
    metric_inputs = {
        "accuracy": {
            "predictions": predictions,
            "labels": labels
        },
        "logit_diff": {
            "target_logits": IO_logits,
            "foil_logits": S_logits
        }
    }
    
    output_path = Path("runs/ioi-discovery-v1_rev3/scratch/stage0_metric_inputs.json")
    with open(output_path, 'w') as f:
        json.dump(metric_inputs, f, indent=2)
    
    print(f"Saved metric inputs to {output_path}")
    print(f"Accuracy: {sum(predictions) / len(predictions):.4f}")
    print(f"Logit diff mean: {sum(d - s for d, s in zip(IO_logits, S_logits)) / len(IO_logits):.4f}")

if __name__ == "__main__":
    main()
