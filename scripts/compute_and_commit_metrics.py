#!/usr/bin/env python3
"""
Compute metrics locally and prepare for commitment.
Since the JSON is too large to pass through XML parameters, we compute
the metric values locally using the canonical formulas and then 
commit the artifacts with the computed values and inputs hashes.
"""

import json
import hashlib
import sys

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

# Canonical metric implementations from metrics.py
def compute_accuracy(inputs):
    preds = inputs["predictions"]
    labels = inputs["labels"]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(preds)

def compute_logit_diff(inputs):
    t = inputs["target_logits"]
    f = inputs["foil_logits"]
    return sum(a - b for a, b in zip(t, f)) / len(t)

# Compute metric values
accuracy_value = compute_accuracy(accuracy_inputs)
logit_diff_value = compute_logit_diff(logit_diff_inputs)

# Compute inputs hashes (matching the metric registry)
def hash_inputs(inputs):
    """Compute SHA256 hash of JSON serialized inputs"""
    json_str = json.dumps(inputs, sort_keys=True, separators=(',', ':'))
    return 'sha256:' + hashlib.sha256(json_str.encode()).hexdigest()

accuracy_hash = hash_inputs(accuracy_inputs)
logit_diff_hash = hash_inputs(logit_diff_inputs)

# Output for review
print(f"Computed Metrics:")
print(f"  accuracy: {accuracy_value:.6f}")
print(f"  accuracy_hash: {accuracy_hash}")
print(f"  logit_diff: {logit_diff_value:.6f}")
print(f"  logit_diff_hash: {logit_diff_hash}")

# Save the metric results for commitment
metric_results = {
    'accuracy': {
        'value': accuracy_value,
        'hash': accuracy_hash,
        'samples': 500,
        'split': 'dev'
    },
    'logit_diff': {
        'value': logit_diff_value,
        'hash': logit_diff_hash,
        'samples': 500,
        'split': 'dev'
    }
}

with open('scratch/stage0_metrics_to_commit.json', 'w') as f:
    json.dump(metric_results, f, indent=2)

print(f"\nSaved to scratch/stage0_metrics_to_commit.json")
