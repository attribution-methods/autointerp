"""Stage 0: Black-box behavioral signal measurement for IOI.

Measure accuracy (logit(IO) > logit(S)) and mean logit_diff on 500 dev samples.
"""
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from autointerp.tools.model import load_model
from autointerp.benchmarks.ioi import generate_pairs

def main():
    # Load GPT-2 small
    print("Loading model gpt2...")
    handle = load_model("gpt2")
    print(f"Model loaded: {handle.model.config.n_layer} layers, {handle.model.config.n_head} heads/layer")
    
    # Generate 500 dev pairs
    print("\nGenerating 500 IOI dev pairs...")
    pairs = generate_pairs(
        spec=None,  # not needed for just generation
        n_pairs=500,
        seed=42,
        tokenizer=handle.tokenizer,
        split="dev"
    )
    print(f"Generated {len(pairs)} pairs")
    
    # Extract clean prompts and target/foil token IDs
    clean_prompts = [p.clean_prompt for p in pairs]
    io_ids = [p.target_token_id for p in pairs]
    s_ids = [p.foil_token_id for p in pairs]
    
    print(f"\nExample pair 0:")
    print(f"  Clean prompt: {clean_prompts[0]!r}")
    print(f"  IO token id: {io_ids[0]} ({handle.tokenizer.decode([io_ids[0]])})")
    print(f"  S token id: {s_ids[0]} ({handle.tokenizer.decode([s_ids[0]])})")
    
    # Run forward passes to get logits at the final position
    print("\nRunning forward passes...")
    batch_size = 50
    n_batches = (len(clean_prompts) + batch_size - 1) // batch_size
    
    io_logits_all = []
    s_logits_all = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(clean_prompts))
        batch_prompts = clean_prompts[start_idx:end_idx]
        
        # Tokenize
        tokens = handle.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(handle.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = handle.model(**tokens)
            logits = outputs.logits  # [batch, seq, vocab]
        
        # Extract logits at the final position for IO and S tokens
        final_logits = logits[:, -1, :]  # [batch, vocab]
        
        for i in range(len(batch_prompts)):
            global_idx = start_idx + i
            io_id = io_ids[global_idx]
            s_id = s_ids[global_idx]
            
            io_logit = final_logits[i, io_id].item()
            s_logit = final_logits[i, s_id].item()
            
            io_logits_all.append(io_logit)
            s_logits_all.append(s_logit)
        
        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  Processed {end_idx}/{len(clean_prompts)} samples")
    
    # Convert to numpy arrays
    io_logits_all = np.array(io_logits_all)
    s_logits_all = np.array(s_logits_all)
    
    # Compute accuracy: fraction where logit(IO) > logit(S)
    correct = (io_logits_all > s_logits_all).astype(float)
    accuracy = correct.mean()
    
    # Compute logit_diff: mean(logit(IO) - logit(S))
    logit_diffs = io_logits_all - s_logits_all
    mean_logit_diff = logit_diffs.mean()
    
    print(f"\n=== Results ===")
    print(f"Accuracy (logit(IO) > logit(S)): {accuracy:.4f} ({correct.sum():.0f}/{len(correct)})")
    print(f"Mean logit_diff: {mean_logit_diff:.4f}")
    print(f"Std logit_diff: {logit_diffs.std():.4f}")
    print(f"Min logit_diff: {logit_diffs.min():.4f}")
    print(f"Max logit_diff: {logit_diffs.max():.4f}")
    
    # Save results
    results = {
        "accuracy": float(accuracy),
        "mean_logit_diff": float(mean_logit_diff),
        "std_logit_diff": float(logit_diffs.std()),
        "n_samples": len(pairs),
        "n_correct": int(correct.sum()),
        "io_logits": io_logits_all.tolist(),
        "s_logits": s_logits_all.tolist(),
        "logit_diffs": logit_diffs.tolist()
    }
    
    import json
    with open("runs/ioi-gpt2-small-blind-v1_rev3/scratch/stage0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to scratch/stage0_results.json")
    
    return accuracy, mean_logit_diff

if __name__ == "__main__":
    main()
