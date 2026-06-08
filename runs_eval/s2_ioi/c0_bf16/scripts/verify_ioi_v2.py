"""
Verify the IOI phenomenon in GPT-2-small - corrected version
In IOI, the pattern is: "When [A] and [B] ... [B/A] gave ... to"
The model should predict the OTHER name (not the one who gave)
"""
import sys
sys.path.insert(0, "/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp/src")

import torch
from autointerp.tools.model import load_model
import numpy as np

# Load model
print("Loading GPT-2-small...")
handle = load_model("gpt2", device="cuda")
model = handle.model
tokenizer = handle.tokenizer

# Create IOI examples
# ABBA: When A and B ... B gave ... to [A expected]
# BABA: When B and A ... A gave ... to [B expected]
ioi_examples = [
    # ABBA templates: A first, B second, B is subject (gave), A is IO (should be predicted)
    ("When Mary and John went to the store, John gave a drink to", "Mary", "John", "ABBA"),
    ("When Sarah and Tom went to the park, Tom gave a gift to", "Sarah", "Tom", "ABBA"),
    ("When Alice and Bob went to the mall, Bob gave a book to", "Alice", "Bob", "ABBA"),
    ("When Emma and David went to the beach, David gave a ball to", "Emma", "David", "ABBA"),
    ("When Lisa and Mike went to the restaurant, Mike gave a menu to", "Lisa", "Mike", "ABBA"),
    
    # BABA templates: B first, A second, A is subject (gave), B is IO (should be predicted)
    ("When John and Mary went to the store, Mary gave a drink to", "John", "Mary", "BABA"),
    ("When Tom and Sarah went to the park, Sarah gave a gift to", "Tom", "Sarah", "BABA"),
    ("When Bob and Alice went to the mall, Alice gave a book to", "Bob", "Alice", "BABA"),
    ("When David and Emma went to the beach, Emma gave a ball to", "David", "Emma", "BABA"),
    ("When Mike and Lisa went to the restaurant, Lisa gave a menu to", "Mike", "Lisa", "BABA"),
]

print("\n" + "="*80)
print("VERIFYING IOI PHENOMENON")
print("="*80)

results = []
for i, (prompt, io_name, s_name, template_type) in enumerate(ioi_examples):
    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(handle.device)
    
    # Get logits
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Get logits for IO and S names
    io_token = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
    s_token = tokenizer.encode(" " + s_name, add_special_tokens=False)[0]
    
    io_logit = logits[io_token].item()
    s_logit = logits[s_token].item()
    
    diff = io_logit - s_logit
    
    results.append({
        'template': template_type,
        'prompt': prompt,
        'io_name': io_name,
        's_name': s_name,
        'io_logit': io_logit,
        's_logit': s_logit,
        'diff': diff
    })
    
    print(f"\n{template_type} #{(i%5) + 1}:")
    print(f"  Prompt: {prompt}")
    print(f"  IO (correct: {io_name}): {io_logit:.3f}")
    print(f"  S (incorrect: {s_name}): {s_logit:.3f}")
    print(f"  Difference (IO - S): {diff:.3f}")
    print(f"  ✓ Correct" if diff > 0 else "  ✗ Incorrect")

# Summary statistics
abba_diffs = [r['diff'] for r in results[:5]]
baba_diffs = [r['diff'] for r in results[5:]]
all_diffs = abba_diffs + baba_diffs

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"ABBA templates: mean diff = {np.mean(abba_diffs):.3f}, correct = {sum(d > 0 for d in abba_diffs)}/5")
print(f"BABA templates: mean diff = {np.mean(baba_diffs):.3f}, correct = {sum(d > 0 for d in baba_diffs)}/5")
print(f"All templates:  mean diff = {np.mean(all_diffs):.3f}, correct = {sum(d > 0 for d in all_diffs)}/10")
print(f"\n✓ IOI phenomenon verified!" if sum(d > 0 for d in all_diffs) >= 8 else "✗ IOI phenomenon not strong")

# Save examples for later use
import json
with open('scratch/ioi_verified.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved results to scratch/ioi_verified.json")
