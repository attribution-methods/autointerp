"""Worked-example discovery evaluators (the reward functions).

These are validated, GPU-backed reward functions for ``discover_features`` —
point at them as ``module:evaluate`` (or copy/adapt for a new model+behavior;
see ``../evaluator_template.py``). They are *examples*, not generic library
code: each is specific to a (model, behavior, substrate) — that specificity is
exactly why the reward can't be one-size-fits-all.

- ``ioi_heads:evaluate`` — GPT-2-small, rank attention heads for IOI, reward =
  mean-ablation logit-diff recovery. Validated: weak baseline ~0.31 →
  DLA-oracle ~0.60; an LLM hill-climb reached ~0.63 and recovered the canonical
  name-mover heads.
- ``surprise:evaluate`` — Qwen2.5-1.5B, rank heads for the semantic-surprise
  contrast (data/semantic_surprise_pairs). Validated: ~0.61 → ~0.94.

Heavy imports (torch / transformer_lens) stay inside functions so importing
this package is cheap.
"""

__all__: list[str] = []
