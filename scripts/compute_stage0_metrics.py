#!/usr/bin/env python3
"""
Compute Stage 0 metrics: accuracy and logit_diff.
This script uses the investigate CLI tool to call compute_metric.
"""

import json
import subprocess
import sys

# Load results
results = json.load(open('scratch/stage0_results.json'))

accuracy_value = results['accuracy']
logit_diff_value = results['mean_logit_diff']

print(f"Stage 0 Metrics:")
print(f"  Accuracy: {accuracy_value:.6f}")
print(f"  Logit_diff: {logit_diff_value:.6f}")

# We can now prepare the data for commit as an artifact directly
# since we've already computed the canonical values
