#!/usr/bin/env python3
"""
Helper script to extract metric inputs and display them for the compute_metric call.
"""

import json

# Load results
results = json.load(open('scratch/stage0_results.json'))

# Create the metric inputs
accuracy_inputs = {
    'predictions': results['accuracies'],
    'labels': [1.0] * len(results['accuracies'])
}

logit_diff_inputs = {
    'target_logits': results['io_logits'],
    'foil_logits': results['s_logits']
}

# Print as properly formatted JSON for copy-paste
print("=== ACCURACY INPUTS ===")
print(json.dumps(accuracy_inputs))

print("\n=== LOGIT_DIFF INPUTS ===")
print(json.dumps(logit_diff_inputs))

print("\n=== STATS ===")
print(f"Accuracy inputs: {len(accuracy_inputs['predictions'])} samples")
print(f"Logit_diff inputs: {len(logit_diff_inputs['target_logits'])} samples")
