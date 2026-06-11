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
- `natural-language-autoencoders`: AV/AR residual-stream activation verbalization, reconstruction scoring, and hypothesis validation.

Add a new skill when a method has distinct triggering conditions, failure modes, or procedures. Add a new tool when the operation needs deterministic code or repeated execution.

## Adding A Skill

1. Create `skills/<skill-name>/SKILL.md` with YAML frontmatter. The
   frontmatter `name` must exactly match the folder name, and `description`
   should be long enough for the agent to decide when to use the skill.
2. Add `skills/<skill-name>/agents/openai.yaml` with a display name and short
   interface description.
3. Update this index with a one-line summary.
4. If the skill corresponds to existing deterministic helper functions, add a
   `SKILL_TO_TOOLS` entry in `src/autointerp/tools/registry.py`. Add new tool
   code only when the operation needs repeatable execution, data validation, or
   heavy computation.
5. Run `python scripts/validate_skills.py` and the local tests before opening a
   PR.
