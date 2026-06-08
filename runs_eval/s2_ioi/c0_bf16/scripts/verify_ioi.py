"""
Verify the IOI phenomenon in GPT-2-small
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

# Create IOI examples - both ABBA and BABA templates
ioi_examples = [
    # ABBA templates (A appears first, then B, then B again, answer is A)
    "When Mary and John went to the store, John gave a drink to",
    "When Sarah and Tom went to the park, Tom gave a gift to",
    "When Alice and Bob went to the mall, Bob gave a book to",
    "When Emma and David went to the beach, David gave a ball to",
    "When Lisa and Mike went to the restaurant, Mike gave a menu to",
    
    # BABA templates (B appears first, then A, then A again, answer is B)
    "When John and Mary went to the store, Mary gave a drink to",
    "When Tom and Sarah went to the park, Sarah gave a gift to",
    "When Bob and Alice went to the mall, Alice gave a book to",
    "When David and Emma went to the beach, Emma gave a ball to",
    "When Mike and Lisa went to the restaurant, Lisa gave a menu to",
]

# Corresponding IO (indirect object) and S (subject) names
io_names = ["Mary", "Sarah", "Alice", "Emma", "Lisa"] * 2
s_names = ["John", "Tom", "Bob", "David", "Mike"] * 2

print("\n" + "="*80)
print("VERIFYING IOI PHENOMENON")
print("="*80)

results = []
for i, prompt in enumerate(ioi_examples):
    io_name = io_names[i]
    s_name = s_names[i]
    
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
    
    template_type = "ABBA" if i < 5 else "BABA"
    results.append({
        'template': template_type,
        'io_logit': io_logit,
        's_logit': s_logit,
        'diff': diff
    })
    
    print(f"\n{template_type} #{i%5 + 1}:")
    print(f"  Prompt: {prompt}")
    print(f"  IO ({io_name}): {io_logit:.3f}")
    print(f"  S ({s_name}): {s_logit:.3f}")
    print(f"  Difference (IO - S): {diff:.3f}")
    print(f"  ✓ IO preferred" if diff > 0 else "  ✗ S preferred")

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
data = {
    'prompts': ioi_examples,
    'io_names': io_names,
    's_names': s_names,
    'results': results
}
with open('scratch/ioi_examples.json', 'w') as f:
    json.dump(data, f, indent=2)

print("\nSaved examples to scratch/ioi_examples.json")
