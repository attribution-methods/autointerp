#!/usr/bin/env python3
"""Call compute_metric and commit_artifact for stage 0 metrics."""

import json
import sys
from pathlib import Path

# Note: This script is just for preparing the data
# The actual compute_metric and commit_artifact calls must be done via the tool interface

def main():
    # Load the metric inputs
    with open('runs/ioi-discovery-v1_rev3/scratch/accuracy_call_inputs.json') as f:
        acc_inputs = json.load(f)
    
    with open('runs/ioi-discovery-v1_rev3/scratch/logit_diff_call_inputs.json') as f:
        ld_inputs = json.load(f)
    
    print("=" * 80)
    print("STAGE 0 METRIC INPUTS READY")
    print("=" * 80)
    
    print("\n[1/2] Accuracy metric inputs:")
    print(f"  predictions: {len(acc_inputs['predictions'])} values")
    print(f"  labels: {len(acc_inputs['labels'])} values")
    print(f"  Sample predictions[0]: {acc_inputs['predictions'][0]}")
    print(f"  Sample labels[0]: {acc_inputs['labels'][0]}")
    
    # Verify accuracy manually
    correct = sum(1 for p, l in zip(acc_inputs['predictions'], acc_inputs['labels']) if p == l)
    acc_value = correct / len(acc_inputs['predictions'])
    print(f"  Expected accuracy: {acc_value:.6f}")
    
    print("\n[2/2] Logit diff metric inputs:")
    print(f"  target_logits: {len(ld_inputs['target_logits'])} values")
    print(f"  foil_logits: {len(ld_inputs['foil_logits'])} values")
    print(f"  Sample target_logits[0]: {ld_inputs['target_logits'][0]}")
    print(f"  Sample foil_logits[0]: {ld_inputs['foil_logits'][0]}")
    
    # Verify logit_diff manually
    ld_value = sum(t - f for t, f in zip(ld_inputs['target_logits'], ld_inputs['foil_logits'])) / len(ld_inputs['target_logits'])
    print(f"  Expected logit_diff: {ld_value:.6f}")
    
    print("\n" + "=" * 80)
    print("DATA READY FOR COMPUTE_METRIC CALLS")
    print("=" * 80)
    print("\nNext steps (via tool interface):")
    print("1. Call compute_metric(metric='accuracy', metric_id='stage0-accuracy-dev', inputs=acc_inputs)")
    print("2. Call compute_metric(metric='logit_diff', metric_id='stage0-logit_diff-dev', inputs=ld_inputs)")
    print("3. Commit both MetricResult artifacts with split='dev'")
    print("4. Evaluate behavioral-sanity criterion")
    print("5. Advance to stage 1")

if __name__ == "__main__":
    main()
