---
name: circuit-discovery
description: Iterative hill-climbing search for a component/feature ranking algorithm that maximizes a causal reward (combined_auc_k). Use when the search space of attribution/ranking methods is large enough that hand-picking one site at a time is too slow, and you want a sub-agent to propose, evaluate, and refine candidate scoring algorithms automatically.
---

# Circuit Discovery (hill-climbing sub-agent)

Delegate the inner "which ranking algorithm finds the causal sites" search to an
iterative sub-agent, instead of hand-coding one attribution method. The
sub-agent proposes a `score(...)` function, the harness evaluates it against a
reward, the best candidate is kept, and the loop climbs.

## When to use

- You have a behavior and contrast pairs, and want to *rank* components
  (attention heads, MLP neurons) or features (SAE latents) by causal importance.
- Trying attribution methods one at a time by hand is the bottleneck.

Use cheaper localization first (`black-box-auditing`, `activation-patching`,
`logit-lens`) when a single concrete site is already in view — discovery pays off
when the *method* is uncertain, not the site.

## How it works

The Tier-2 `discover_features` tool runs the loop in
`src/autointerp/pipelines/investigation/discovery/`:

1. Seeds `algorithm_v1.py` from `algorithm_template.py` (the `score(...)`
   contract returning `list[Candidate]` sorted by `abs(score)`).
2. Each iteration proposes one improved candidate (same LiteLLM model the
   runtime uses), evaluates it via `harness.py`, and keeps it only if the reward
   strictly improves (hill-climb with early-stop `patience`).
3. Returns the best candidate's reward + `top_features`.

Reward is `combined_auc_k = 0.5*(mean_ablation_auc_k + mean_steering_auc_k)` by
default — necessity (ablation) + sufficiency (steering) over a top-K sweep.

## Usage

`discover_features` is only allowed in a stage whose `tools` list includes it,
and its reward must be one of that stage's pre-registered `metrics`. Tune the
search via the stage's optional `DiscoveryConfig` (`max_iterations`, `patience`,
`top_k`, `k_grid`, `evaluator`).

Smoke test with no model or GPU:

```bash
PYTHONPATH=src python -m autointerp.pipelines.investigation.discovery.harness \
  --algorithm src/autointerp/pipelines/investigation/discovery/algorithm_template.py \
  --output /tmp/out.json --dry-run
```

For a real run, supply an evaluator (`--evaluator module:attr`, or
`DiscoveryConfig.evaluator`) that loads the model, applies the candidate's
ranking, ablates/steers the top-K over the K-grid, and returns the AUC payload.

## Discipline

`discover_features` is **advisory** — it commits nothing. Record evidence the
normal way: `commit_artifact` for `CandidateSite`/`FeatureFinding`, and
`compute_metric` + `commit_artifact` for the reward on a **heldout** split. A
high discovery-time reward on `dev` is a lead, not a result.
