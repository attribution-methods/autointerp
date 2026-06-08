#!/usr/bin/env python3
"""Stage 0: Black-box IOI evaluation - FIXED."""

import sys
import json
import random
from pathlib import Path
import torch
import numpy as np

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

from autointerp.tools.model import load_model

NAMES = ["John", "Mary", "Tom", "Sarah", "James", "Kate", "Robert", "Lisa", 
         "Michael", "Emma", "David", "Sophia", "Chris", "Anna", "Mark", "Grace"]
PLACES = ["park", "store", "mall", "office", "library", "cafe", "gym", "school"]
OBJECTS = ["book", "pen", "bag", "phone", "gift", "letter", "card", "toy"]

def generate_ioi_dataset(n_samples=500, seed=42):
    """Generate IOI prompts.
    
    ABBA: "When A and B went to the park, A gave a book to"
          Answer should be B (indirect object). A appears twice (subject).
    BABA: "When A and B went to the park, B gave a book to"  
          Answer should be A (indirect object). B appears twice (subject).
    """
    random.seed(seed)
    np.random.seed(seed)
    
    samples = []
    n_abba = n_samples // 2
    n_baba = n_samples - n_abba
    
    # ABBA: A appears twice, answer is B
    for _ in range(n_abba):
        a, b = random.sample(NAMES, 2)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        prompt = f"When {a} and {b} went to the {place}, {a} gave a {obj} to"
        samples.append({
            "prompt": prompt,
            "io_name": b,  # B is indirect object
            "s_name": a,   # A is subject (appears twice)
            "template": "ABBA"
        })
    
    # BABA: B appears twice, answer is A
    for _ in range(n_baba):
        a, b = random.sample(NAMES, 2)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        prompt = f"When {a} and {b} went to the {place}, {b} gave a {obj} to"
        samples.append({
            "prompt": prompt,
            "io_name": a,  # A is indirect object
            "s_name": b,   # B is subject (appears twice)
            "template": "BABA"
        })
    
    random.shuffle(samples)
    return samples

def compute_logit_diff_batch(handle, samples, batch_size=20):
    results = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = [s["prompt"] for s in batch]
        
        tokens = handle.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(handle.input_device())
        
        with torch.no_grad():
            outputs = handle.model(**tokens)
            logits = outputs.logits
        
        seq_lens = tokens["attention_mask"].sum(dim=1)
        
        for j, sample in enumerate(batch):
            final_pos = seq_lens[j] - 1
            final_logits = logits[j, final_pos]
            
            io_token = handle.tokenizer.encode(" " + sample["io_name"], add_special_tokens=False)[0]
            s_token = handle.tokenizer.encode(" " + sample["s_name"], add_special_tokens=False)[0]
            
            io_logit = final_logits[io_token].item()
            s_logit = final_logits[s_token].item()
            logit_diff = io_logit - s_logit
            
            results.append({
                "prompt": sample["prompt"],
                "io_name": sample["io_name"],
                "s_name": sample["s_name"],
                "template": sample["template"],
                "io_logit": io_logit,
                "s_logit": s_logit,
                "logit_diff": logit_diff,
                "correct": logit_diff > 0
            })
    
    return results

def main():
    print("Stage 0: Black-box IOI evaluation (FIXED)")
    print("=" * 60)
    
    samples = generate_ioi_dataset(n_samples=500, seed=42)
    print(f"Generated {len(samples)} samples")
    
    out_dir = Path("scratch")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "ioi_samples_dev.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Loading GPT-2 small...")
    handle = load_model("gpt2", device="cuda", dtype=torch.float32)
    print(f"Model: {handle.model_id} (layers={handle.n_layers})")
    
    print(f"Computing logit_diff...")
    results = compute_logit_diff_batch(handle, samples, batch_size=20)
    
    with open(out_dir / "ioi_results_dev.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logit_diffs = [r["logit_diff"] for r in results]
    correct = [r["correct"] for r in results]
    
    mean_logit_diff = np.mean(logit_diffs)
    std_logit_diff = np.std(logit_diffs)
    accuracy = np.mean(correct)
    
    print("\nRESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({sum(correct)}/{len(correct)})")
    print(f"Mean logit_diff: {mean_logit_diff:.4f} ± {std_logit_diff:.4f}")
    print(f"Min: {min(logit_diffs):.4f}, Max: {max(logit_diffs):.4f}")
    
    if accuracy >= 0.95:
        print("\n✓ Accuracy >= 0.95 - behavioral sanity check PASSES")
    elif accuracy >= 0.9:
        print("\n✓ Accuracy >= 0.9 - above abort threshold")
    else:
        print("\n⚠️  Accuracy < 0.9 - ABORT condition")
    
    # Check clean-corrupt gap
    if mean_logit_diff >= 2.0:
        print(f"✓ Logit diff {mean_logit_diff:.2f} >= 2.0 - strong signal")
    else:
        print(f"⚠️  Logit diff {mean_logit_diff:.2f} < 2.0 - weak signal")
    
    summary = {
        "n_samples": len(results),
        "accuracy": float(accuracy),
        "mean_logit_diff": float(mean_logit_diff),
        "std_logit_diff": float(std_logit_diff),
        "min_logit_diff": float(min(logit_diffs)),
        "max_logit_diff": float(max(logit_diffs))
    }
    
    with open(out_dir / "stage0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    handle.cleanup()
    print("\nDone!")

if __name__ == "__main__":
    main()
