# patch_effect_recovery

**Family:** causal · **Range:** [0, 1] · **Direction:** higher is better

Fraction of the clean-vs-corrupted behavioral gap recovered when patching k
candidate components from the clean run into the corrupted run.

`recovery = (patched - corrupt) / (clean - corrupt)`

## When to use
- Localizing causal sites with activation patching (Meng et al., Wang et al.).
- Validating circuit candidates: a circuit with high recovery at low k is a
  good circuit.

## Pitfalls
- Denominator can be small or noisy if the clean/corrupt gap is weak — confirm
  baseline `logit_diff` is solid before citing recovery numbers.
- Recovery > 1 is possible (over-patching); clamp or report explicitly.


## Contract
- family: causal
- value_range: [0.0, 1.0]
- direction: higher-is-better
- requires_inputs: ['clean_metric', 'corrupt_metric', 'patched_metric']
