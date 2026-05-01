---
name: linear-probes
description: Linear and lightweight nonlinear probing of activations. Use for testing whether a label, concept, behavior state, answer, or hidden variable is decodable from a layer, token position, component, or cached activation set.
---

# Linear Probes

## Workflow

1. Build a balanced labeled dataset with train/test separation by prompt family when possible.
2. Extract activations at a fixed layer, component, and token position.
3. Train a linear probe first, then a small MLP only if linear decoding fails.
4. Report cross-validation accuracy, F1, AUROC, and class balance.
5. Analyze the learned direction and test it with steering or ablation if claiming mechanistic relevance.

## Tools

Use `../../src/autointerp/tools/probes.py`:

```python
from autointerp.tools.probes import train_probe, probe_direction

probe, scaler, report = train_probe(acts.numpy(), labels, probe_type="linear")
direction = probe_direction(probe, scaler)
```

## Evidence

Probes show availability of information, not necessarily model use. Stronger evidence comes from causal tests on the probe direction or sites that support the probe.

## Cautions

Avoid leakage from near-duplicate prompts, labels encoded in formatting, or evaluating on the same distribution used to design prompts.
