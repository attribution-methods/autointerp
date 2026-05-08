#!/usr/bin/env python3
"""Stage 0: Black-box evaluation of IOI on GPT-2-small dev split.

Computes:
- accuracy: fraction where logit(IO) > logit(S) at the final position
- logit_diff: mean(logit(IO) - logit(S)) across all examples

Both metrics are SCALARS as required by the spec.
"""

import sys
import json
import torch
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
    print(f"Generated {len(pairs)} pairs")
    
    # Run forward passes on clean prompts
    print("Running forward passes on clean prompts...")
    clean_prompts = [p.clean_prompt for p in pairs]
    io_token_ids = [p.target_token_id for p in pairs]
    s_token_ids = [p.foil_token_id for p in pairs]
    
    # Process in batches to avoid OOM
    batch_size = 32
    all_io_logits = []
    all_s_logits = []
    
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
            
            # Extract logits at the final position for IO and S tokens
            for j, (io_id, s_id) in enumerate(zip(batch_io_ids, batch_s_ids)):
                final_logits = logits[j, -1, :]  # [vocab]
                io_logit = final_logits[io_id].cpu().item()
                s_logit = final_logits[s_id].cpu().item()
                all_io_logits.append(io_logit)
                all_s_logits.append(s_logit)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Processed {i + len(batch_prompts)}/{len(clean_prompts)} prompts")
    
    # Convert to numpy arrays
    io_logits = np.array(all_io_logits)
    s_logits = np.array(all_s_logits)
    
    # Compute metrics
    logit_diffs = io_logits - s_logits
    accuracy = float(np.mean(logit_diffs > 0))
    mean_logit_diff = float(np.mean(logit_diffs))
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy (IO > S):  {accuracy:.4f}")
    print(f"Mean logit_diff:    {mean_logit_diff:.4f}")
    print(f"Std logit_diff:     {np.std(logit_diffs):.4f}")
    print(f"Min logit_diff:     {np.min(logit_diffs):.4f}")
    print(f"Max logit_diff:     {np.max(logit_diffs):.4f}")
    print("="*60)
    
    # Save detailed results to scratch
    results = {
        "accuracy": accuracy,
        "mean_logit_diff": mean_logit_diff,
        "std_logit_diff": float(np.std(logit_diffs)),
        "min_logit_diff": float(np.min(logit_diffs)),
        "max_logit_diff": float(np.max(logit_diffs)),
        "n_samples": len(pairs),
        "io_logits_shape": io_logits.shape,
        "s_logits_shape": s_logits.shape
    }
    
    output_path = run_dir / "scratch" / "stage0_blackbox_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")
    
    # Also save the pairs for reuse
    pairs_path = run_dir / "scratch" / "ioi_dev_pairs.json"
    pairs_data = [
        {
            "pair_id": p.pair_id,
            "clean_prompt": p.clean_prompt,
            "corrupted_prompt": p.corrupted_prompt,
            "target_token_id": p.target_token_id,
            "foil_token_id": p.foil_token_id,
            "prediction_position": p.prediction_position,
            "metadata": p.metadata
        }
        for p in pairs
    ]
    with open(pairs_path, "w") as f:
        json.dump(pairs_data, f, indent=2)
    print(f"IOI pairs saved to {pairs_path}")

if __name__ == "__main__":
    main()
