#!/usr/bin/env python3
"""
Generate IOI dataset according to spec.json specifications.
"""

import json
import random
from pathlib import Path
from itertools import combinations

# Single-token names that GPT-2 tokenizes with a leading space
NAMES = [
    " John", " Mary", " Tom", " Sarah", " James", " Kate", 
    " Robert", " Lisa", " David", " Emma", " Michael", " Anna",
    " William", " Susan", " Richard", " Emily", " Joseph", " Laura"
]

PLACES = [
    " park", " store", " school", " office", " library", " mall",
    " museum", " beach", " restaurant", " cafe"
]

OBJECTS = [
    " book", " pen", " gift", " letter", " key", " phone",
    " bag", " watch", " ring", " card"
]

def generate_ioi_prompt(name_a, name_b, place, obj, template_type):
    """
    Generate an IOI prompt.
    
    template_type: 'ABBA' or 'BABA'
    ABBA: "When {A} and {B} went to the {place}, {A} gave a {object} to" -> answer is B (non-repeated)
    BABA: "When {A} and {B} went to the {place}, {B} gave a {object} to" -> answer is A (non-repeated)
    
    Returns: (prompt, IO_token, S_token)
        - IO_token: the indirect object (correct answer) - the name NOT in subject position
        - S_token: the subject (the name that appears in "gave" position)
    """
    if template_type == 'ABBA':
        prompt = f"When{name_a} and{name_b} went to the{place},{name_a} gave a{obj} to"
        io_token = name_b  # B is the indirect object (non-repeated at END, correct answer)
        s_token = name_a   # A is the subject (appears in "gave" position)
    else:  # BABA
        prompt = f"When{name_a} and{name_b} went to the{place},{name_b} gave a{obj} to"
        io_token = name_a  # A is the indirect object (non-repeated at END, correct answer)
        s_token = name_b   # B is the subject (appears in "gave" position)
    
    return prompt, io_token, s_token

def generate_abc_corrupt_prompt(name_a, name_b, name_c, place, obj, template_type):
    """
    Generate ABC-corrupted IOI prompt.
    Replace the first occurrence of the name that appears twice with a novel name C.
    
    For ABBA (A...A): replace first A with C -> "When C and B went to the place, A gave a object to"
    For BABA (A...B...B): replace first B with C -> "When A and C went to the place, B gave a object to"
    
    This breaks the duplicate-name structure while keeping END-position context unchanged.
    
    Returns: (prompt, IO_token, S_token)
    """
    if template_type == 'ABBA':
        # Original: When A and B..., A gave...
        # Corrupt: When C and B..., A gave...
        prompt = f"When{name_c} and{name_b} went to the{place},{name_a} gave a{obj} to"
        io_token = name_b  # Target should still be B (the non-giver)
        s_token = name_a   # Subject is still A
    else:  # BABA
        # Original: When A and B..., B gave...
        # Corrupt: When A and C..., B gave...
        prompt = f"When{name_a} and{name_c} went to the{place},{name_b} gave a{obj} to"
        io_token = name_a  # Target should still be A (the non-giver)
        s_token = name_b   # Subject is still B
    
    return prompt, io_token, s_token

def generate_dataset(n_samples, split_name, seed, output_dir):
    """Generate IOI dataset with balanced ABBA/BABA templates."""
    random.seed(seed)
    
    # Generate name pairs
    name_pairs = list(combinations(NAMES, 2))
    random.shuffle(name_pairs)
    
    samples = []
    n_abba = n_samples // 2
    n_baba = n_samples - n_abba
    
    for i in range(n_samples):
        template_type = 'ABBA' if i < n_abba else 'BABA'
        
        # Select names (cycle through pairs if needed)
        pair_idx = i % len(name_pairs)
        name_a, name_b = name_pairs[pair_idx]
        
        # Select random place and object
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        
        # Generate clean prompt
        clean_prompt, io_token, s_token = generate_ioi_prompt(name_a, name_b, place, obj, template_type)
        
        # Generate corrupted prompt (ABC)
        # Select a third name different from A and B
        available_names = [n for n in NAMES if n not in [name_a, name_b]]
        name_c = random.choice(available_names)
        corrupt_prompt, _, _ = generate_abc_corrupt_prompt(name_a, name_b, name_c, place, obj, template_type)
        
        sample = {
            'idx': i,
            'prompt': clean_prompt,
            'corrupt_prompt': corrupt_prompt,
            'IO': io_token,
            'S': s_token,
            'template_type': template_type,
            'names': {'A': name_a, 'B': name_b, 'C': name_c},
            'place': place,
            'object': obj
        }
        
        samples.append(sample)
    
    # Save dataset
    output_path = output_dir / f"{split_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} samples for {split_name} split")
    print(f"Saved to {output_path}")
    
    # Print statistics
    n_abba_actual = sum(1 for s in samples if s['template_type'] == 'ABBA')
    n_baba_actual = sum(1 for s in samples if s['template_type'] == 'BABA')
    print(f"  ABBA: {n_abba_actual}, BABA: {n_baba_actual}")
    
    return samples

def main():
    output_dir = Path("datasets") / "ioi-abba-baba-gpt2-v1"
    
    # Generate dev split (500 samples, seed=42)
    print("Generating dev split...")
    random.seed(42)
    name_pairs_dev = list(combinations(NAMES, 2))
    random.shuffle(name_pairs_dev)
    dev_pairs = name_pairs_dev[:500]  # Reserve name pairs for dev
    
    dev_samples = generate_dataset(500, "dev", seed=42, output_dir=output_dir)
    
    # Generate heldout split (100 samples, different seed, disjoint names)
    print("\nGenerating heldout split...")
    heldout_samples = generate_dataset(100, "heldout", seed=43, output_dir=output_dir)
    
    print("\nDataset generation complete!")
    print(f"Dev samples: {len(dev_samples)}")
    print(f"Heldout samples: {len(heldout_samples)}")

if __name__ == "__main__":
    main()
