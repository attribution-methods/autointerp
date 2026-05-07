"""
Stage 2: Activation analysis - mean ablate top heads and measure drops.
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

def main():
    print("Loading GPT-2-small...")
    handle = load_model('gpt2')
    
    print("Loading IOI dev dataset...")
    clean_examples = load_ioi_dataset(split='dev', n_samples=500)
    clean_prompts = [ex['prompt'] for ex in clean_examples]
    
    # Get IO and S token IDs
    io_token_ids = [handle.tokenizer.encode(ex['io_name'], add_special_tokens=False)[0] 
                    for ex in clean_examples]
    s_token_ids = [handle.tokenizer.encode(ex['s_name'], add_special_tokens=False)[0] 
                   for ex in clean_examples]
    io_token_ids = torch.tensor(io_token_ids, device=handle.device)
    s_token_ids = torch.tensor(s_token_ids, device=handle.device)
    
    # Load top heads from localization
    with open('scratch/stage1_joint_recovery.json') as f:
        localization_results = json.load(f)
    
    top_3_heads = localization_results['top_3_heads']
    print(f"\\nTop-3 heads from localization: {top_3_heads}")
    
    # First compute baseline logit_diff on clean
    print("\\nComputing baseline logit_diff on clean...")
    tokenized = handle.tokenizer(clean_prompts, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = handle.model(
            input_ids=tokenized['input_ids'].to(handle.device),
            attention_mask=tokenized['attention_mask'].to(handle.device)
        )
        clean_logits = outputs.logits
    
    # Extract final position logits
    clean_lD = []
    for i in range(len(clean_prompts)):
        seq_len = tokenized['attention_mask'][i].sum().item()
        last_pos = seq_len - 1
        io_logit = clean_logits[i, last_pos, io_token_ids[i]].item()
        s_logit = clean_logits[i, last_pos, s_token_ids[i]].item()
        clean_lD.append(io_logit - s_logit)
    
    baseline_logit_diff = np.mean(clean_lD)
    print(f"Baseline logit_diff: {baseline_logit_diff:.4f}")
    
    # Cache clean head z for ablation
    print("\\nCaching head activations for ablation...")
    clean_cache = cache_head_z(handle, clean_prompts)
    mean_z = mean_head_z(clean_cache)
    
    # Ablate each head individually
    ablation_results = []
    
    for layer, head in top_3_heads:
        print(f"\\nAblating L{layer}H{head}...")
        
        ablated_logits = mean_ablate_heads(
            handle, clean_prompts, [(layer, head)], mean_z
        )
        
        # Compute logit_diff on ablated outputs
        ablated_lD = logit_diff(ablated_logits, io_token_ids, s_token_ids).mean().item()
        
        # Compute ablation drop
        drop = baseline_logit_diff - ablated_lD
        drop_pct = (drop / baseline_logit_diff) * 100
        
        print(f"  Ablated logit_diff: {ablated_lD:.4f}")
        print(f"  Drop: {drop:.4f} ({drop_pct:.1f}%)")
        
        ablation_results.append({
            'layer': layer,
            'head': head,
            'ablated_logit_diff': ablated_lD,
            'drop': drop,
            'drop_percentage': drop_pct
        })
    
    # Find the head with maximum drop
    max_drop_head = max(ablation_results, key=lambda x: x['drop'])
    
    print("\\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Baseline logit_diff: {baseline_logit_diff:.4f}")
    print(f"\\nIndividual head ablation drops:")
    for r in ablation_results:
        print(f"  L{r['layer']}H{r['head']}: {r['drop']:.4f} ({r['drop_percentage']:.1f}%)")
    
    print(f"\\nHead with maximum drop: L{max_drop_head['layer']}H{max_drop_head['head']}")
    print(f"  Drop: {max_drop_head['drop']:.4f}")
    
    # Save results
    results = {
        'baseline_logit_diff': baseline_logit_diff,
        'ablation_results': ablation_results,
        'max_drop_head': max_drop_head
    }
    
    with open('scratch/stage2_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nSaved results to scratch/stage2_ablation_results.json")
    
    return results

if __name__ == "__main__":
    main()
