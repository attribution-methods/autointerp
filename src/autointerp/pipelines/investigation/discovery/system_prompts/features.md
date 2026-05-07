## Substrate: features (decomposition=${decomposition})

The "feature" your algorithm ranks is a **learned direction** in some
decomposition of activations — typically an SAE feature (one row of an
SAE encoder weight, paired with its decoder column) or a transcoder /
probe direction. The harness performs feature-projection / steering
interventions based on the decomposition family. Return
`Candidate(kind="sae_feature", layer=<int>, idx=<feature_idx>,
score=<float>)` rows (or analogous for other decompositions).

## What `loader` exposes
- `.model` — HuggingFace causal LM.
- `.saes` — `dict[layer_idx -> SAE]`: encoder, decoder, bias for each
  hooked layer.
- `.tokenizer`.
- `.hook_context()` — context manager that installs SAE forward hooks
  and exposes `.get_activations(layer)` to retrieve per-token feature
  activations on the most recent forward pass.

## Algorithm Ideas to Try (feature substrate)
- **Activation difference**: per (layer, feature), score =
  `mean(|f_clean − f_corrupted|)`. Cheap, no gradient. Mirrors the
  strongest baseline in the circuitbreaker reference.
- **Directional patching**: for each feature, replace its activation
  with the corrupted value at a specific token position; measure the
  output-logit shift.
- **Token-specific EAP-IG**: integrated-gradient attribution but
  retaining `token_pos`; allows the validator to ablate or steer only
  at the causally relevant position.
- **Layer-normalized scoring**: normalize EAP-IG / activation-diff
  scores within each layer to counteract scale drift.
- **Multi-step patching**: patch features at multiple positions
  simultaneously and measure the joint effect.

## Pitfalls
- Steering with very-high-norm features takes the residual out of
  distribution — perplexity blows up. Report perplexity in metadata so
  off-distribution moves don't masquerade as good steering.
- A constant steering coefficient α is part of the protocol; if you
  change α between candidates the AUC numbers are not comparable.
- Don't reuse the same pairs for ranker selection and reward
  measurement — split-tag your output if the spec uses heldout pairs
  for the criterion.
