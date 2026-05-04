"""Tool registry metadata for agent scaffolds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ToolSpec:
    name: str
    module: str
    function: str
    description: str


TOOLS: List[ToolSpec] = [
    ToolSpec(
        "sample",
        "autointerp.tools.blackbox",
        "sample",
        "Send chat prompts with optional prefill.",
    ),
    ToolSpec(
        "complete-text",
        "autointerp.tools.blackbox",
        "complete_text",
        "Run raw text completion.",
    ),
    ToolSpec(
        "sample-user-initial",
        "autointerp.tools.blackbox",
        "sample_user_initial",
        "Sample an initial user turn.",
    ),
    ToolSpec(
        "sample-user-followup",
        "autointerp.tools.blackbox",
        "sample_user_followup",
        "Sample a follow-up user turn.",
    ),
    ToolSpec(
        "extract-tagged-json",
        "autointerp.tools.blackbox",
        "extract_tagged_json",
        "Parse tagged JSON scenarios.",
    ),
    ToolSpec(
        "capture-activations",
        "autointerp.tools.activations",
        "capture_activations",
        "Extract activations at layers/components.",
    ),
    ToolSpec(
        "cache-components",
        "autointerp.tools.activations",
        "cache_components",
        "Cache multiple activation sites.",
    ),
    ToolSpec(
        "contrastive-direction",
        "autointerp.tools.vectors",
        "contrastive_direction",
        "Compute mean-difference directions.",
    ),
    ToolSpec(
        "baseline-subtracted-direction",
        "autointerp.tools.vectors",
        "baseline_subtracted_direction",
        "Compute baseline-subtracted directions.",
    ),
    ToolSpec(
        "batch-contrastive-directions",
        "autointerp.tools.vectors",
        "batch_contrastive_directions",
        "Compute many contrastive directions.",
    ),
    ToolSpec(
        "cosine-similarity",
        "autointerp.tools.vectors",
        "cosine_similarity",
        "Compare vector directions.",
    ),
    ToolSpec(
        "generate-with-steering",
        "autointerp.tools.generation",
        "generate_with_steering",
        "Apply activation steering during generation.",
    ),
    ToolSpec(
        "generate-with-multi-steering",
        "autointerp.tools.generation",
        "generate_with_multi_steering",
        "Apply multiple steering vectors.",
    ),
    ToolSpec(
        "logit-lens",
        "autointerp.tools.lenses",
        "logit_lens",
        "Decode intermediate token predictions.",
    ),
    ToolSpec(
        "direct-logit-attribution",
        "autointerp.tools.lenses",
        "direct_logit_attribution",
        "Score direct residual-to-logit contributions.",
    ),
    ToolSpec(
        "patch-generation",
        "autointerp.tools.patching",
        "patch_generation",
        "Patch activations and generate.",
    ),
    ToolSpec(
        "ablate-generation",
        "autointerp.tools.patching",
        "ablate_generation",
        "Ablate a layer/component during generation.",
    ),
    ToolSpec(
        "sweep-patch-sites",
        "autointerp.tools.patching",
        "sweep_patch_sites",
        "Sweep patch sites across layers/components.",
    ),
    ToolSpec(
        "cache-head-z",
        "autointerp.tools.head_patching",
        "cache_head_z",
        "Cache per-head pre-W_O activations (z) at chosen layers.",
    ),
    ToolSpec(
        "mean-head-z",
        "autointerp.tools.head_patching",
        "mean_head_z",
        "Average cached head z over the batch to build a mean-ablation baseline.",
    ),
    ToolSpec(
        "run-with-head-patches",
        "autointerp.tools.head_patching",
        "run_with_head_patches",
        "Forward pass with arbitrary (layer, head, position) z overrides.",
    ),
    ToolSpec(
        "mean-ablate-heads",
        "autointerp.tools.head_patching",
        "mean_ablate_heads",
        "Mean-ablate one or more (layer, head) sites and return logits.",
    ),
    ToolSpec(
        "head-patch-sweep",
        "autointerp.tools.head_patching",
        "head_patch_sweep",
        "Sweep clean->corrupt patching across (layer, head) pairs.",
    ),
    ToolSpec(
        "path-patch",
        "autointerp.tools.head_patching",
        "path_patch",
        "Single-step path patching: sender's direct effect with others frozen to corrupt.",
    ),
    ToolSpec(
        "logit-diff",
        "autointerp.tools.head_patching",
        "logit_diff",
        "Per-row logit(IO)-logit(S) metric for paired-token sweeps.",
    ),
    ToolSpec(
        "train-probe",
        "autointerp.tools.probes",
        "train_probe",
        "Train a linear or MLP probe.",
    ),
    ToolSpec(
        "gradient-attribution",
        "autointerp.tools.attribution",
        "next_token_gradient_x_activation",
        "Score token positions by gradient times activation.",
    ),
    ToolSpec(
        "attribution-patch-score",
        "autointerp.tools.attribution",
        "attribution_patch_score",
        "Estimate patching effects with first-order attribution.",
    ),
    ToolSpec(
        "rank-attributions",
        "autointerp.tools.attribution",
        "rank_attributions",
        "Rank attribution score maps.",
    ),
    ToolSpec(
        "top-sae-features",
        "autointerp.tools.sae",
        "top_features",
        "Rank SAE features by activation.",
    ),
    ToolSpec(
        "attach-sae-labels",
        "autointerp.tools.sae",
        "attach_labels",
        "Attach labels to SAE feature rows.",
    ),
    ToolSpec(
        "filter-semantic-features",
        "autointerp.tools.sae",
        "filter_semantic_features",
        "Drop low-semantic SAE features.",
    ),
    ToolSpec(
        "format-feature-cluster",
        "autointerp.tools.sae",
        "format_feature_cluster",
        "Format SAE feature clusters.",
    ),
    ToolSpec(
        "rank-circuit-sites",
        "autointerp.tools.circuits",
        "rank_sites",
        "Rank candidate circuit sites.",
    ),
    ToolSpec(
        "combine-circuit-scores",
        "autointerp.tools.circuits",
        "combine_scores",
        "Combine evidence scores across methods.",
    ),
    ToolSpec(
        "formula-score",
        "autointerp.tools.circuits",
        "formula_score",
        "Compute an aggregate circuit evidence score.",
    ),
    ToolSpec(
        "summarize-circuit",
        "autointerp.tools.circuits",
        "summarize_circuit",
        "Summarize ranked circuit sites.",
    ),
]

SKILL_TO_TOOLS: Dict[str, List[str]] = {
    "black-box-auditing": [
        "sample",
        "complete-text",
        "sample-user-initial",
        "sample-user-followup",
        "extract-tagged-json",
    ],
    "activation-cache": ["capture-activations", "cache-components"],
    "contrastive-directions": [
        "contrastive-direction",
        "baseline-subtracted-direction",
        "batch-contrastive-directions",
        "cosine-similarity",
    ],
    "activation-steering": ["generate-with-steering", "generate-with-multi-steering"],
    "logit-lens": ["logit-lens", "direct-logit-attribution"],
    "tuned-lens": ["logit-lens"],
    "activation-patching": [
        "patch-generation",
        "ablate-generation",
        "sweep-patch-sites",
        "cache-head-z",
        "mean-head-z",
        "run-with-head-patches",
        "mean-ablate-heads",
        "head-patch-sweep",
        "path-patch",
        "logit-diff",
    ],
    "attribution-patching": ["attribution-patch-score", "rank-attributions"],
    "linear-probes": ["train-probe"],
    "gradient-attribution": ["gradient-attribution"],
    "sparse-autoencoders": [
        "top-sae-features",
        "attach-sae-labels",
        "filter-semantic-features",
        "format-feature-cluster",
    ],
    "attention-heads": [
        "capture-activations",
        "ablate-generation",
        "sweep-patch-sites",
        "cache-head-z",
        "mean-head-z",
        "run-with-head-patches",
        "mean-ablate-heads",
        "head-patch-sweep",
        "path-patch",
        "logit-diff",
    ],
    "qk-ov-decomposition": ["capture-activations", "direct-logit-attribution"],
    "mlp-neuron-analysis": [
        "capture-activations",
        "direct-logit-attribution",
        "ablate-generation",
    ],
    "circuit-tracing": [
        "rank-circuit-sites",
        "combine-circuit-scores",
        "formula-score",
        "summarize-circuit",
    ],
    "causal-validation": [
        "patch-generation",
        "ablate-generation",
        "sweep-patch-sites",
        "mean-ablate-heads",
        "path-patch",
    ],
    "activation-oracles": ["capture-activations", "sample", "extract-tagged-json"],
}
