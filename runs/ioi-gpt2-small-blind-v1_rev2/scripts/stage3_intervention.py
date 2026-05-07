"""
Stage 3: Intervention - measure circuit faithfulness by ablating complement.
"""
import sys
sys.path.insert(0, '../../src')

import torch
import json
import numpy as np
from autointerp.tools.model import load_model
from autointerp.tools.head_patching import (
    cache_head_z, mean_head_z, mean_ablate_heads, logit_diff
)

def load_ioi_dataset(split="dev", n_samples=500):
    """Load IOI dataset."""
    names = [' John', ' Mary', ' Tom', ' Sarah', ' James', ' Kate', ' Robert', ' Lisa',
             ' Michael', ' Emily', ' David', ' Anna', ' Chris', ' Laura', ' Mark', ' Emma']
    places = [' park', ' store', ' mall', ' school', ' beach', ' museum', ' office', ' library']
    objects = [' book', ' pen', ' gift', ' letter', ' toy', ' drink', ' card', ' bag']
    
    examples = []
    np.random.seed(42 if split == 'dev' else 43)
    
    for i in range(n_samples):
        name_indices = np.random.choice(len(names), 2, replace=False)
        A, B = names[name_indices[0]], names[name_indices[1]]
        place = np.random.choice(places)
        obj = np.random.choice(objects)
        
        if i % 2 == 0:
            prompt = f'When{A} and{B} went to the{place},{B} gave a{obj} to'
            io_name, s_name, template = A, B, 'BABA'
        else:
            prompt = f'When{A} and{B} went to the{place},{A} gave a{obj} to'
            io_name, s_name, template = B, A, 'ABBA'
        
        examples.append({
            'prompt': prompt, 'io_name': io_name, 's_name': s_name, 
            'template': template, 'A': A, 'B': B, 'place': place, 'object': obj
        })
    return examples

def create_abc_corruption(examples, tokenizer):
    """Create ABC-corrupted prompts."""
    all_names = [' John', ' Mary', ' Tom', ' Sarah', ' James', ' Kate', ' Robert', ' Lisa',
                 ' Michael', ' Emily', ' David', ' Anna', ' Chris', ' Laura', ' Mark', ' Emma']
    
    corrupted = []
    for ex in examples:
        available = [n for n in all_names if n not in [ex['A'], ex['B']]]
        C = np.random.choice(available)
        
        if ex['template'] == 'BABA':
            corrupt_prompt = f"When{ex['A']} and{C} went to the{ex['place']},{ex['B']} gave a{ex['object']} to"
        else:
            corrupt_prompt = f"When{C} and{ex['B']} went to the{ex['place']},{ex['A']} gave a{ex['object']} to"
        
        corrupted.append({'prompt': corrupt_prompt})
    return corrupted

def main():
    print("Loading GPT-2-small...")
    handle = load_model('gpt2')
    
    print("Loading IOI dev dataset...")
    clean_examples = load_ioi_dataset(split='dev', n_samples=500)
    corrupt_examples = create_abc_corruption(clean_examples, handle.tokenizer)
    
    clean_prompts = [ex['prompt'] for ex in clean_examples]
    corrupt_prompts = [ex['prompt'] for ex in corrupt_examples]
    
    # Get IO and S token IDs
    io_token_ids = [handle.tokenizer.encode(ex['io_name'], add_special_tokens=False)[0] 
                    for ex in clean_examples]
    s_token_ids = [handle.tokenizer.encode(ex['s_name'], add_special_tokens=False)[0] 
                   for ex in clean_examples]
    io_token_ids = torch.tensor(io_token_ids, device=handle.device)
    s_token_ids = torch.tensor(s_token_ids, device=handle.device)
    
    # Load the circuit (top-3 heads from localization)
    with open('scratch/stage1_joint_recovery.json') as f:
        localization_results = json.load(f)
    
    circuit_heads = localization_results['top_3_heads']
    circuit_heads = [(l, h) for l, h in circuit_heads]
    print(f"\\nCircuit heads: {circuit_heads}")
    
    # GPT-2-small has 12 layers and 12 heads per layer
    n_layers = 12
    n_heads = 12
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    
    # Complement of circuit (all heads NOT in the circuit)
    complement_heads = [head for head in all_heads if head not in circuit_heads]
    print(f"\\nComplement heads: {len(complement_heads)} heads (ablating these)")
    
    # Compute clean logit_diff
    print("\\nComputing clean logit_diff...")
    tokenized = handle.tokenizer(clean_prompts, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = handle.model(
            input_ids=tokenized['input_ids'].to(handle.device),
            attention_mask=tokenized['attention_mask'].to(handle.device)
        )
        clean_logits = outputs.logits
    
    clean_lD = []
    for i in range(len(clean_prompts)):
        seq_len = tokenized['attention_mask'][i].sum().item()
        last_pos = seq_len - 1
        io_logit = clean_logits[i, last_pos, io_token_ids[i]].item()
        s_logit = clean_logits[i, last_pos, s_token_ids[i]].item()
        clean_lD.append(io_logit - s_logit)
    
    clean_logit_diff = np.mean(clean_lD)
    print(f"Clean logit_diff: {clean_logit_diff:.4f}")
    
    # Compute corrupt logit_diff
    print("\\nComputing corrupt logit_diff...")
    tokenized_corrupt = handle.tokenizer(corrupt_prompts, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = handle.model(
            input_ids=tokenized_corrupt['input_ids'].to(handle.device),
            attention_mask=tokenized_corrupt['attention_mask'].to(handle.device)
        )
        corrupt_logits = outputs.logits
    
    corrupt_lD = []
    for i in range(len(corrupt_prompts)):
        seq_len = tokenized_corrupt['attention_mask'][i].sum().item()
        last_pos = seq_len - 1
        io_logit = corrupt_logits[i, last_pos, io_token_ids[i]].item()
        s_logit = corrupt_logits[i, last_pos, s_token_ids[i]].item()
        corrupt_lD.append(io_logit - s_logit)
    
    corrupt_logit_diff = np.mean(corrupt_lD)
    print(f"Corrupt logit_diff: {corrupt_logit_diff:.4f}")
    
    # Cache clean head activations for ablation
    print("\\nCaching head activations...")
    clean_cache = cache_head_z(handle, clean_prompts)
    mean_z = mean_head_z(clean_cache)
    
    # Ablate the COMPLEMENT of the circuit (circuit-only run)
    print(f"\\nAblating {len(complement_heads)} heads (complement of circuit)...")
    circuit_only_logits = mean_ablate_heads(
        handle, clean_prompts, complement_heads, mean_z
    )
    
    # Compute logit_diff on circuit-only run
    circuit_only_lD = logit_diff(circuit_only_logits, io_token_ids, s_token_ids).mean().item()
    print(f"Circuit-only logit_diff: {circuit_only_lD:.4f}")
    
    # Compute faithfulness
    faithfulness = (circuit_only_lD - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)
    
    print("\\n" + "="*60)
    print("FAITHFULNESS RESULTS:")
    print("="*60)
    print(f"Clean logit_diff: {clean_logit_diff:.4f}")
    print(f"Corrupt logit_diff: {corrupt_logit_diff:.4f}")
    print(f"Circuit-only logit_diff: {circuit_only_lD:.4f}")
    print(f"Faithfulness: {faithfulness:.4f}")
    print(f"\\nFormula: ({circuit_only_lD:.4f} - {corrupt_logit_diff:.4f}) / ({clean_logit_diff:.4f} - {corrupt_logit_diff:.4f})")
    
    # Also report recovery from circuit restoration (for patch_effect_recovery metric)
    # This is the same as what we computed in stage 1, but let's include it for completeness
    recovery = (circuit_only_lD - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)
    
    # Save results
    results = {
        'circuit_heads': circuit_heads,
        'clean_logit_diff': clean_logit_diff,
        'corrupt_logit_diff': corrupt_logit_diff,
        'circuit_only_logit_diff': circuit_only_lD,
        'faithfulness': faithfulness,
        'recovery': recovery
    }
    
    with open('scratch/stage3_faithfulness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nSaved results to scratch/stage3_faithfulness_results.json")
    
    return results

if __name__ == "__main__":
    main()
