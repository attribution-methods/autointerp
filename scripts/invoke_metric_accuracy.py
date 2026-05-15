#!/usr/bin/env python3
"""
Helper to invoke compute_metric with accuracy data.
This reads the pre-computed inputs and prints them in a format suitable for the tool.
"""

import json
import sys

# Load the accuracy inputs
with open('scratch/accuracy_inputs_final.json', 'r') as f:
    accuracy_inputs = json.load(f)

# Verify counts
print(f"Accuracy inputs loaded:", file=sys.stderr)
print(f"  Predictions: {len(accuracy_inputs['predictions'])}", file=sys.stderr)
print(f"  Labels: {len(accuracy_inputs['labels'])}", file=sys.stderr)

# Print the JSON to stdout for use in compute_metric call
# The tool should be able to parse this directly
print(json.dumps(accuracy_inputs))
