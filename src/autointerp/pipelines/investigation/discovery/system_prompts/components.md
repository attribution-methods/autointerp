## Substrate: components (${component_kinds})

The "feature" your algorithm ranks is a **component** in the model's
computation graph — an attention head, MLP neuron, MLP layer, or
residual-stream layer site. The harness performs the appropriate
ablation / patching intervention based on the component kind. Return
`Candidate(kind="<kind>", layer=<int>, idx=<int>, score=<float>)` rows.

## What `loader` exposes
- `.model` — HuggingFace causal LM (`AutoModelForCausalLM`).
- `.tokenizer` — its tokenizer (left-padded for batched inputs).
- `.device`, `.dtype`, `.n_layers`, `.n_heads`, `.d_model`.
- `.input_device()` — where to put input tensors.

The cheapest way to ablate / patch components correctly is the helpers
in `autointerp.tools.head_patching` (real, generic across HF causal LMs):

```python
from autointerp.tools.head_patching import (
    cache_head_z,           # cache per-head z [B, S, H, d_head]
    mean_head_z,             # build the mean-ablation baseline
    mean_ablate_heads,       # mean-ablate (layer, head) sites
    run_with_head_patches,   # forward pass with arbitrary HeadPatch overrides
    head_patch_sweep,        # full L x H clean->corrupt patching sweep
    path_patch,              # single-step path patching
    logit_diff,              # batched logit_diff metric
)
```

For MLP neurons / layers / residual sites, use
`autointerp.tools.patching` and `autointerp.tools.activations`.

## Algorithm Ideas to Try (head substrate)
- **Mean-z-diff**: per (layer, head), score = `mean(|z_clean − z_corrupt|)`
  at the prediction position. Cheap, gradient-free, surprisingly strong
  on contrastive tasks.
- **Head-patch-sweep ranking**: run `head_patch_sweep` once, rank heads
  by per-head `patch_effect_recovery`. Strong baseline; almost free
  given a single sweep.
- **Path-patching attribution**: for each candidate sender head, run
  `path_patch` to its OUT projection only; rank by direct effect.
  Catches "writes to logits" heads that simple ablation misses.
- **Gradient × activation**: cache activations and per-head gradients of
  `logit_diff(target, foil)`; rank heads by `mean(g · z)`. First-order
  attribution; faster than full path patching.
- **Layer-normalized score**: rank heads by their per-layer-normalized
  attribution to counteract scale differences across layers.
- **Position-specific scoring**: keep the `token_pos` field on
  `Candidate` and ablate only at the END position; often improves the
  reward because the IOI signal lives at one position.

## Pitfalls
- Mean-ablating the entire prompt (default) hurts more than necessary.
  Pass `positions=[-1]` (or use `Candidate.token_pos`) to ablate only at
  the prediction position when the candidate is position-specific.
- Sweeping all 144 head positions on GPT-2-small is fast (<60s on an
  H100); don't pre-prune to a guessed shortlist.
- Beware sign: by convention `Candidate.score > 0` should mean
  "promotes the clean (target) behavior." Heads that *inhibit* the
  behavior also matter — use absolute value for ranking, but keep sign
  in the metadata.
