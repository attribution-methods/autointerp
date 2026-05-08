# logit_diff

**Family:** behavioral · **Range:** unbounded · **Direction:** higher is better

`logit(target_token) - logit(foil_token)` at the prediction position. Standard
behavioral signal for contrastive tasks (IOI, factual recall, role assignment).

## When to use
- You have a clean target/foil token pair per prompt.
- You want a continuous signal rather than discrete accuracy.

## Pitfalls
- Magnitude depends on model temperature scale; don't compare across models
  without normalizing.
- Mean across prompts can hide a heavy tail — also report per-prompt distribution.


## Contract
- family: behavioral
- value_range: [-∞, ∞]
- direction: higher-is-better
- requires_inputs: ['target_logits', 'foil_logits']
