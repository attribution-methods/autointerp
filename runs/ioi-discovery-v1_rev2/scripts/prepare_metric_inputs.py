#!/usr/bin/env python3
"""Prepare metric inputs for compute_metric from the blackbox results."""

import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs

def main():
    run_dir = Path(__file__).parent.parent
    spec_path = run_dir / "spec.json"
    
    # Load spec
    with open(spec_path) as f:
        spec = json.load(f)
    
    print("Loading GPT-2-small model...")
    import torch
    handle = load_model("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate IOI dev pairs
    print(f"Generating {spec['dataset']['n_samples']} IOI pairs for dev split...")
    pairs = generate_pairs(
        spec,
        n_pairs=spec['dataset']['n_samples'],
        seed=spec['dataset']['seed'],
        tokenizer=handle.tokenizer,
        split="dev"
    )
    
    # Run forward passes on clean prompts
    print("Running forward passes on clean prompts...")
    clean_prompts = [p.clean_prompt for p in pairs]
    io_token_ids = [p.target_token_id for p in pairs]
    s_token_ids = [p.foil_token_id for p in pairs]
    
    # Process in batches
    batch_size = 32
    all_io_logits = []
    all_s_logits = []
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(clean_prompts), batch_size):
            batch_prompts = clean_prompts[i:i+batch_size]
            batch_io_ids = io_token_ids[i:i+batch_size]
            batch_s_ids = s_token_ids[i:i+batch_size]
            
            # Tokenize
            inputs = handle.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(handle.device)
            
            # Forward pass
            outputs = handle.model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab]
            
            # Extract logits and predictions at the final position
            for j, (io_id, s_id) in enumerate(zip(batch_io_ids, batch_s_ids)):
                final_logits = logits[j, -1, :]  # [vocab]
                predicted_id = final_logits.argmax().cpu().item()
                io_logit = final_logits[io_id].cpu().item()
                s_logit = final_logits[s_id].cpu().item()
                
                all_io_logits.append(io_logit)
                all_s_logits.append(s_logit)
                all_predictions.append(predicted_id)
    
    # Convert to lists for JSON serialization
    io_logits_list = [float(x) for x in all_io_logits]
    s_logits_list = [float(x) for x in all_s_logits]
    predictions_list = [int(x) for x in all_predictions]
    labels_list = [int(x) for x in io_token_ids]
    
    # Save for compute_metric
    metric_inputs = {
        "accuracy": {
            "predictions": predictions_list,
            "labels": labels_list
        },
        "logit_diff": {
            "target_logits": io_logits_list,
            "foil_logits": s_logits_list
        }
    }
    
    output_path = run_dir / "scratch" / "stage0_metric_inputs.json"
    with open(output_path, "w") as f:
        json.dump(metric_inputs, f, indent=2)
    
    print(f"\nMetric inputs saved to {output_path}")
    print(f"  accuracy: {len(predictions_list)} predictions, {len(labels_list)} labels")
    print(f"  logit_diff: {len(io_logits_list)} target_logits, {len(s_logits_list)} foil_logits")

if __name__ == "__main__":
    main()
