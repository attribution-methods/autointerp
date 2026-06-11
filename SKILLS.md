# Skill Index

The first toolkit pass includes these skills:

- `black-box-auditing`: chat, prefill, user-role sampling, text completion, scaffolded batches.
- `activation-cache`: model loading, tokenization, component addressing, activation extraction.
- `contrastive-directions`: mean-difference directions, baselines, vector geometry.
- `activation-steering`: steering with contrastive or learned directions.
- `activation-patching`: activation replacement, causal tracing, component patching.
- `attribution-patching`: first-order patching estimates for component triage.
- `gradient-attribution`: saliency, gradient times activation, integrated-gradient style checks.
- `logit-lens`: intermediate residual predictions and KL-based filtering.
- `tuned-lens`: learned translators for calibrated intermediate predictions.
- `sparse-autoencoders`: SAE feature lookup, activation ranking, feature labeling.
- `predictive-concept-decoders`: Transluce-style sparse concept bottleneck decoders for behavior-predictive activation readouts.
- `linear-probes`: linear and small nonlinear probes with cross-validation.
- `attention-heads`: head discovery, attention pattern inspection, head ablation.
- `qk-ov-decomposition`: QK attention logic and OV vocabulary/output analysis.
- `mlp-neuron-analysis`: MLP/neuron contribution and activation analysis.
- `circuit-tracing`: graph-style circuit discovery and formula ranking.
- `causal-validation`: ablations, swaps, counterfactual prompts, held-out validation.
- `activation-oracles`: natural-language activation verbalizers and evidence filtering.

Add a new skill when a method has distinct triggering conditions, failure modes, or procedures. Add a new tool when the operation needs deterministic code or repeated execution.
