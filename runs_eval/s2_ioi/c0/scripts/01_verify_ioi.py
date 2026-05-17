"""
Verify the IOI phenomenon in GPT-2-small and analyze basic logit differences.
"""
import sys
sys.path.insert(0, '/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp')

import torch
from autointerp.tools.model import load_model
from autointerp.tools.lenses import top_tokens
import numpy as np

# Load GPT-2-small
print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

# Define IOI test prompts (ABBA and BABA templates)
test_prompts = [
    # ABBA templates
    "When John and Mary went to the store, Mary gave a drink to",
    "When Sarah and Tom went to the park, Tom gave a ball to",
    "When Alice and Bob went to the beach, Bob gave a gift to",
    "When Emma and Jack went to the mall, Jack gave a book to",
    
    # BABA templates  
    "When Mary and John went to the store, Mary gave a drink to",
    "When Tom and Sarah went to the park, Tom gave a ball to",
    "When Bob and Alice went to the beach, Bob gave a gift to",
    "When Jack and Emma went to the mall, Jack gave a book to",
]

# Corresponding name pairs (A is indirect object, B is subject)
name_pairs = [
    ("John", "Mary"),   # ABBA
    ("Sarah", "Tom"),    # ABBA
    ("Alice", "Bob"),    # ABBA
    ("Emma", "Jack"),    # ABBA
    ("John", "Mary"),    # BABA
    ("Sarah", "Tom"),    # BABA
    ("Alice", "Bob"),    # BABA
    ("Emma", "Jack"),    # BABA
]

print("\n" + "="*80)
print("VERIFYING IOI PHENOMENON")
print("="*80)

results = []

for i, (prompt, (name_a, name_b)) in enumerate(zip(test_prompts, name_pairs)):
    template_type = "ABBA" if i < 4 else "BABA"
    
    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(handle.device)
    
    # Get logits
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Get token IDs for names (handle potential multi-token names)
    name_a_ids = tokenizer.encode(" " + name_a, add_special_tokens=False)
    name_b_ids = tokenizer.encode(" " + name_b, add_special_tokens=False)
    
    # For simplicity, use first token of each name
    name_a_id = name_a_ids[0]
    name_b_id = name_b_ids[0]
    
    logit_a = logits[name_a_id].item()
    logit_b = logits[name_b_id].item()
    logit_diff = logit_a - logit_b
    
    results.append({
        'template': template_type,
        'prompt': prompt,
        'name_a': name_a,
        'name_b': name_b,
        'logit_a': logit_a,
        'logit_b': logit_b,
        'logit_diff': logit_diff
    })
    
    print(f"\n{template_type} Template #{i+1}:")
    print(f"Prompt: {prompt}")
    print(f"Name A (indirect object): {name_a}, Logit: {logit_a:.3f}")
    print(f"Name B (subject): {name_b}, Logit: {logit_b:.3f}")
    print(f"Logit Difference (A - B): {logit_diff:.3f}")
    
    # Show top predictions
    top_5 = top_tokens(handle, logits, top_k=5)
    print(f"Top 5 predictions: {top_5}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

abba_diffs = [r['logit_diff'] for r in results if r['template'] == 'ABBA']
baba_diffs = [r['logit_diff'] for r in results if r['template'] == 'BABA']

print(f"\nABBA templates:")
print(f"  Mean logit difference (A - B): {np.mean(abba_diffs):.3f}")
print(f"  Std: {np.std(abba_diffs):.3f}")
print(f"  All positive: {all(d > 0 for d in abba_diffs)}")

print(f"\nBABA templates:")
print(f"  Mean logit difference (A - B): {np.mean(baba_diffs):.3f}")
print(f"  Std: {np.std(baba_diffs):.3f}")
print(f"  All positive: {all(d > 0 for d in baba_diffs)}")

print(f"\nOverall:")
all_diffs = abba_diffs + baba_diffs
print(f"  Mean logit difference (A - B): {np.mean(all_diffs):.3f}")
print(f"  Std: {np.std(all_diffs):.3f}")
print(f"  Success rate (A > B): {sum(d > 0 for d in all_diffs)}/{len(all_diffs)}")

# Save results
import json
with open('/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/runs_eval/s2_ioi/c0/scratch/ioi_verification.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to scratch/ioi_verification.json")
