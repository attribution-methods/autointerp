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

The Tier-2 `discover_features` tool runs the generic engine
(`run_hillclimb`) in `src/autointerp/pipelines/investigation/discovery/`:

1. Seeds a candidate from `algorithm_template.py` (the `score(...)` contract
   returning `list[Candidate]` sorted by `abs(score)`).
2. Each round proposes `n_subagents` candidates concurrently — either a single
   LLM completion or a tool-using subagent (`run_agent_turn`), same LiteLLM
   model path as the runtime — seeded from the current top-K **archive**, and
   evaluates each via `harness.py`.
3. Keeps a top-K archive (population), early-stops on `patience`, and re-checks
   the best across `seeds` to flag flukes (`best_reward_std`).
4. Returns the best candidate's reward + `top_features` + archive. Token/cost
   spend is bridged into the run's `budget_consumed`.

Reward is `combined_auc_k = 0.5*(mean_ablation_auc_k + mean_steering_auc_k)` by
default — necessity (ablation) + sufficiency (steering) over a top-K sweep.

## Usage

`discover_features` is only allowed in a stage whose `tools` list includes it,
and its reward must be one of that stage's pre-registered `metrics`. Tune the
search via the stage's frozen `DiscoveryConfig` (`max_iterations`, `patience`,
`n_subagents`, `archive_size`, `propose_mode`, `seeds`, `model`, `max_tokens`,
`top_k`, `k_grid`, `objective`, `evaluator`).

Smoke test with no model or GPU:

```bash
PYTHONPATH=src python -m autointerp.pipelines.investigation.discovery.harness \
  --algorithm src/autointerp/pipelines/investigation/discovery/algorithm_template.py \
  --output /tmp/out.json --dry-run
```

For a real run you must supply an **evaluator** — the reward function that
loads the model, applies the candidate's ranking, intervenes (ablate/steer the
top-K over the K-grid), and returns
`{mean_ablation_auc_k, mean_steering_auc_k, top_features}` from
`evaluate(score_fn, *, top_k, k_grid, seed)`. Wire it via
`DiscoveryConfig.evaluator` or `--evaluator`, in either form:

- `module:attr` — a shipped/importable module, e.g.
  `autointerp.pipelines.investigation.discovery.evaluators.ioi_heads:evaluate`
  (GPT-2 IOI) or `...evaluators.surprise:evaluate` (Qwen surprise).
- `path/to/file.py:attr` — an evaluator YOU write (e.g. in the run's
  `scripts/`). The evaluator is model/behavior/substrate-specific, so for a new
  investigation, **write your own**: copy
  `src/autointerp/pipelines/investigation/discovery/evaluator_template.py`,
  fill in the TODOs, and point `--evaluator scripts/my_eval.py:evaluate`.

The template bakes in the correct intervention boilerplate (per-layer hook
closures, ablate the right positions, return the modified output) and a
**mandatory sanity check** that the intervention actually moves the metric.
The harness also flags any real evaluator that returns exactly 0.0 as a likely
silent no-op — fix the evaluator, not the ranking.

## Discipline

`discover_features` is **advisory** — it commits nothing. Record evidence the
normal way: `commit_artifact` for `CandidateSite`/`FeatureFinding`, and
`compute_metric` + `commit_artifact` for the reward on a **heldout** split. A
high discovery-time reward on `dev` is a lead, not a result.
