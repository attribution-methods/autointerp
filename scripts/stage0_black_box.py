#!/usr/bin/env python3
"""
Stage 0: Black-box behavioral sanity check.
Compute accuracy and logit_diff for clean IOI prompts.
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# IOI dataset constants (from Wang et al. 2022 / priors)
NAMES = [' John', ' Mary', ' Tom', ' Sarah', ' James', ' Kate', ' Robert', ' Lisa',
         ' Michael', ' Jennifer', ' William', ' Patricia', ' David', ' Barbara',
         ' Richard', ' Susan', ' Joseph', ' Jessica', ' Thomas', ' Angela']

PLACES = [' park', ' store', ' beach', ' museum', ' restaurant', ' library', ' school']
OBJECTS = [' book', ' pen', ' apple', ' hat', ' ball', ' cup', ' flower']

CLEAN_TEMPLATES = [
    "When {A} and {B} went to the {place}, {B} gave a {object} to",  # BABA: repeat B
    "When {A} and {B} went to the {place}, {A} gave a {object} to",   # ABBA: repeat A
]

def generate_ioi_sample(idx, names, places, objects, templates, seed=42):
    """Generate a single IOI sample deterministically."""
    torch.manual_seed(seed + idx)
    
    template_idx = idx % len(templates)
    template = templates[template_idx]
    
    # Deterministic name selection
    name_a_idx = (idx * 2) % len(names)
    name_b_idx = (idx * 3) % len(names)
    while name_b_idx == name_a_idx:
        name_b_idx = (name_b_idx + 1) % len(names)
    
    place_idx = (idx * 5) % len(places)
    obj_idx = (idx * 7) % len(objects)
    
    A = names[name_a_idx]
    B = names[name_b_idx]
    place = places[place_idx]
    obj = objects[obj_idx]
    
    prompt = template.format(A=A, B=B, place=place, object=obj)
    
    # Determine target (IO) and foil (S) names
    if template_idx == 0:  # BABA: repeat B, IO is A
        io_name = A
        s_name = B
    else:  # ABBA: repeat A, IO is B
        io_name = B
        s_name = A
    
    return {
        'prompt': prompt,
        'io_name': io_name,
        's_name': s_name,
        'template_idx': template_idx
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Generate 500 IOI dev samples (n_samples from spec, seed=42)
    print("Generating 500 IOI dev samples with seed=42...")
    samples = []
    for idx in range(500):
        sample = generate_ioi_sample(idx, NAMES, PLACES, OBJECTS, CLEAN_TEMPLATES, seed=42)
        samples.append(sample)
    
    print("Running inference on clean prompts to collect logits...")
    
    io_logits_list = []
    s_logits_list = []
    accuracies = []
    
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            prompt = sample['prompt']
            io_name = sample['io_name']
            s_name = sample['s_name']
            
            # Tokenize
            tokens = tokenizer.encode(prompt)
            tokens_tensor = torch.tensor([tokens], device=device)
            
            # Forward pass: get logits
            outputs = model(tokens_tensor)
            logits = outputs.logits[0, -1, :]  # logits at final position
            
            # Get IO and S token IDs
            io_token_id = tokenizer.encode(io_name)[0]
            s_token_id = tokenizer.encode(s_name)[0]
            
            io_logit = logits[io_token_id].item()
            s_logit = logits[s_token_id].item()
            
            io_logits_list.append(io_logit)
            s_logits_list.append(s_logit)
            
            # Binary accuracy: IO logit > S logit
            acc = 1.0 if io_logit > s_logit else 0.0
            accuracies.append(acc)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/500 samples")
    
    # Compute statistics
    logit_diffs = [io - s for io, s in zip(io_logits_list, s_logits_list)]
    accuracy = sum(accuracies) / len(accuracies)
    mean_logit_diff = sum(logit_diffs) / len(logit_diffs)
    
    print(f"\n=== Stage 0 Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean IO logit: {sum(io_logits_list) / len(io_logits_list):.4f}")
    print(f"Mean S logit: {sum(s_logits_list) / len(s_logits_list):.4f}")
    print(f"Mean logit_diff (IO - S): {mean_logit_diff:.4f}")
    print(f"Min logit_diff: {min(logit_diffs):.4f}")
    print(f"Max logit_diff: {max(logit_diffs):.4f}")
    
    # Save results for metric computation
    results = {
        'io_logits': io_logits_list,
        's_logits': s_logits_list,
        'logit_diffs': logit_diffs,
        'accuracies': accuracies,
        'accuracy': accuracy,
        'mean_logit_diff': mean_logit_diff,
        'samples': samples
    }
    
    output_path = Path('scratch/stage0_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
