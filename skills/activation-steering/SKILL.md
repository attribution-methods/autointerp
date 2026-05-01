---
name: activation-steering
description: Activation steering with concept, contrastive, persona, refusal, honesty, or prefill-derived directions. Use to test whether adding a vector changes generations, elicits hidden behavior, suppresses behavior, or provides causal evidence that a direction controls a model property.
---

# Activation Steering

## Workflow

1. Compute or load a direction with `contrastive-directions`.
2. Choose a layer and strength sweep before inspecting outputs.
3. Generate paired steered and unsteered responses on the same prompts.
4. Evaluate both desired behavior change and collateral damage.
5. Treat steering as causal evidence only if controls and held-out prompts behave as predicted.

## Tools

Use `../../src/autointerp/tools/generation.py`:

```python
from autointerp.tools.generation import generate_with_steering

normal = handle.generate(prompt, max_new_tokens=100)
steered = generate_with_steering(handle, prompt, layer=40, steering_vector=direction, strength=1.25)
```

## Good Controls

Use random directions with matched norm, unrelated concept directions, opposite sign steering, and multiple strengths. Check whether the vector simply increases verbosity or willingness to speculate.

## Cautions

Strong steering can create artifacts that look like evidence. Prefer small sweeps and compare against baseline models or behavior-absent prompts.
