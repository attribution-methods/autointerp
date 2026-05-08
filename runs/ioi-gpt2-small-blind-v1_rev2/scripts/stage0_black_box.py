"""
Stage 0: Black-box IOI behavioral signal measurement.
Measure accuracy and logit_diff on 500 dev samples.
"""
import sys
sys.path.insert(0, "src")

import torch
import json
import numpy as np
from autointerp.tools.model import load_model

def load_ioi_dataset(split="dev", n_samples=500):
    """Load IOI dataset from generated examples."""
    # For this investigation, we need to generate IOI prompts
    # Format: "When {A} and {B} went to the {place}, {B/A} gave a {object} to"
    # IO is the non-repeated name, S is the repeated name
    
    # Single-token names for GPT-2 (with leading space)
    names = [" John", " Mary", " Tom", " Sarah", " James", " Kate", " Robert", " Lisa",
             " Michael", " Emily", " David", " Anna", " Chris", " Laura", " Mark", " Emma"]
    places = [" park", " store", " mall", " school", " beach", " museum", " office", " library"]
    objects = [" book", " pen", " gift", " letter", " toy", " drink", " card", " bag"]
    
    examples = []
    np.random.seed(42 if split == "dev" else 43)
    
    # Generate balanced ABBA/BABA examples
    for i in range(n_samples):
        # Pick two distinct names
        name_indices = np.random.choice(len(names), 2, replace=False)
        A, B = names[name_indices[0]], names[name_indices[1]]
        place = np.random.choice(places)
        obj = np.random.choice(objects)
        
        # Alternate BABA and ABBA
        if i % 2 == 0:
            # BABA: A appears once, B appears twice -> IO=A, S=B
            prompt = f"When{A} and{B} went to the{place},{B} gave a{obj} to"
            io_name = A
            s_name = B
            template = "BABA"
        else:
            # ABBA: B appears once, A appears twice -> IO=B, S=A  
            prompt = f"When{A} and{B} went to the{place},{A} gave a{obj} to"
            io_name = B
            s_name = A
            template = "ABBA"
        
        examples.append({
            "prompt": prompt,
            "io_name": io_name,
            "s_name": s_name,
            "template": template,
            "names": [A, B],
            "place": place,
            "object": obj
        })
    
    return examples

def main():
    print("Loading GPT-2-small model...")
    handle = load_model("gpt2")
    
    print("Loading IOI dev dataset (500 samples)...")
    examples = load_ioi_dataset(split="dev", n_samples=500)
    
    print(f"First 3 examples:")
    for i in range(3):
        ex = examples[i]
        print(f"  {i}: {ex['prompt']}")
        print(f"      IO={ex['io_name']}, S={ex['s_name']}, template={ex['template']}")
    
    # Batch process to get logits
    prompts = [ex["prompt"] for ex in examples]
    
    print("\nTokenizing prompts...")
    tokenized = handle.tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(handle.device)
    attention_mask = tokenized["attention_mask"].to(handle.device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample token length: {input_ids[0].sum()}")
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = handle.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab]
    
    print(f"Logits shape: {logits.shape}")
    
    # Extract logits at final position for IO and S tokens
    io_logits = []
    s_logits = []
    
    for i, ex in enumerate(examples):
        # Find the last non-padding token position
        seq_len = attention_mask[i].sum().item()
        last_pos = seq_len - 1
        
        # Get token IDs for IO and S names
        io_token_id = handle.tokenizer.encode(ex["io_name"], add_special_tokens=False)[0]
        s_token_id = handle.tokenizer.encode(ex["s_name"], add_special_tokens=False)[0]
        
        # Extract logits for these tokens at the last position
        io_logit = logits[i, last_pos, io_token_id].item()
        s_logit = logits[i, last_pos, s_token_id].item()
        
        io_logits.append(io_logit)
        s_logits.append(s_logit)
    
    io_logits = np.array(io_logits)
    s_logits = np.array(s_logits)
    
    # Compute metrics
    logit_diffs = io_logits - s_logits
    mean_logit_diff = logit_diffs.mean()
    accuracy = (logit_diffs > 0).mean()
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Accuracy (IO logit > S logit): {accuracy:.4f} ({int(accuracy*500)}/500)")
    print(f"Mean logit_diff (IO - S): {mean_logit_diff:.4f}")
    print(f"Logit diff std: {logit_diffs.std():.4f}")
    print(f"Logit diff range: [{logit_diffs.min():.4f}, {logit_diffs.max():.4f}]")
    
    # Check abort conditions
    print("\n" + "="*60)
    print("ABORT CONDITION CHECKS:")
    print("="*60)
    if accuracy < 0.9:
        print(f"⚠️  WARNING: Accuracy {accuracy:.4f} < 0.9 (abort threshold)")
    else:
        print(f"✓ Accuracy {accuracy:.4f} >= 0.9")
    
    clean_corrupt_gap = mean_logit_diff  # Approximate for now
    if clean_corrupt_gap < 2.0:
        print(f"⚠️  WARNING: Logit diff {clean_corrupt_gap:.4f} < 2.0 nats (abort threshold)")
    else:
        print(f"✓ Logit diff {clean_corrupt_gap:.4f} >= 2.0 nats")
    
    # Save results
    results = {
        "accuracy": float(accuracy),
        "mean_logit_diff": float(mean_logit_diff),
        "logit_diff_std": float(logit_diffs.std()),
        "n_samples": len(examples),
        "io_logits_summary": {
            "mean": float(io_logits.mean()),
            "std": float(io_logits.std())
        },
        "s_logits_summary": {
            "mean": float(s_logits.mean()),
            "std": float(s_logits.std())
        }
    }
    
    with open("runs/ioi-gpt2-small-blind-v1_rev2/scratch/stage0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to scratch/stage0_results.json")
    
    return results

if __name__ == "__main__":
    main()
