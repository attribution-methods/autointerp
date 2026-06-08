#!/usr/bin/env python3
"""
Stage 0: Compute and commit the accuracy and logit_diff metrics.
"""

import json
import sys
import os

# Add the src directory to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load results
results = json.load(open('scratch/stage0_results.json'))

# Prepare metric inputs
accuracy_inputs = {
    'predictions': results['accuracies'],
    'labels': [1.0] * len(results['accuracies'])
}

logit_diff_inputs = {
    'target_logits': results['io_logits'],
    'foil_logits': results['s_logits']
}

# Compute using the canonical formulas
def compute_accuracy(inputs):
    """From metrics.py: Mean(predictions[i] == labels[i])"""
    preds = inputs["predictions"]
    labels = inputs["labels"]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(preds)

def compute_logit_diff(inputs):
    """From metrics.py: Mean(target_logits[i] - foil_logits[i])"""
    t = inputs["target_logits"]
    f = inputs["foil_logits"]
    return sum(a - b for a, b in zip(t, f)) / len(t)

accuracy_value = compute_accuracy(accuracy_inputs)
logit_diff_value = compute_logit_diff(logit_diff_inputs)

print(f"Computed Metrics:")
print(f"  accuracy: {accuracy_value:.6f}")
print(f"  logit_diff: {logit_diff_value:.6f}")

# Save for later use
metrics_summary = {
    'accuracy': accuracy_value,
    'logit_diff': logit_diff_value,
    'samples_dev': 500,
    'stage': 'black_box'
}

with open('scratch/stage0_metrics_computed.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\nSaved to scratch/stage0_metrics_computed.json")
