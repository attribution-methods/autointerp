#!/usr/bin/env python3
"""Test single head patching to debug the joint recovery issue."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autointerp.tools.model import load_model
from autointerp.tools.head_patching import cache_head_z, run_with_head_patches, HeadPatch, logit_diff
import json

# Load model
handle = load_model("gpt2")
device = handle.device

# Load a few samples
with open("datasets/ioi-abba-baba-gpt2-v1/dev.jsonl") as f:
    samples = [json.loads(line) for line in f][:10]

clean_prompts = [s['prompt'] for s in samples]
corrupt_prompts = [s['corrupt_prompt'] for s in samples]

target_ids = torch.tensor([handle.tokenizer.encode(s['IO'], add_special_tokens=False)[0] for s in samples])
contrast_ids = torch.tensor([handle.tokenizer.encode(s['S'], add_special_tokens=False)[0] for s in samples])

# Cache activations
print("Caching activations...")
clean_cache = cache_head_z(handle, clean_prompts)
corrupt_cache = cache_head_z(handle, corrupt_prompts)

# Compute baselines
print("\nComputing baselines...")
inputs_clean = handle.tokenizer(clean_prompts, return_tensors='pt', padding=True)
inputs_clean = {k: v.to(device) for k, v in inputs_clean.items()}
with torch.no_grad():
    clean_logits = handle.model(**inputs_clean).logits[:, -1, :]
clean_metric = logit_diff(clean_logits, target_ids.to(device), contrast_ids.to(device)).mean().item()

inputs_corrupt = handle.tokenizer(corrupt_prompts, return_tensors='pt', padding=True)
inputs_corrupt = {k: v.to(device) for k, v in inputs_corrupt.items()}
with torch.no_grad():
    corrupt_logits = handle.model(**inputs_corrupt).logits[:, -1, :]
corrupt_metric = logit_diff(corrupt_logits, target_ids.to(device), contrast_ids.to(device)).mean().item()

gap = clean_metric - corrupt_metric

print(f"Clean: {clean_metric:.4f}")
print(f"Corrupt: {corrupt_metric:.4f}")
print(f"Gap: {gap:.4f}")

# Test patching L9H9 (should give ~0.46 recovery)
layer, head = 9, 9
print(f"\nPatching L{layer}H{head}...")

# Get source from clean cache
source = clean_cache[layer][:, -1, head, :]  # [batch, d_head]
patch = HeadPatch(layer=layer, head=head, source=source, positions=[-1])

# Run with patch on corrupt prompts
patched_logits = run_with_head_patches(handle, corrupt_prompts, [patch])
patched_metric = logit_diff(patched_logits, target_ids.to(device), contrast_ids.to(device)).mean().item()

recovery = (patched_metric - corrupt_metric) / gap

print(f"Patched: {patched_metric:.4f}")
print(f"Recovery: {recovery:.4f}")
print(f"Expected: ~0.46")

