---
name: circuit-discovery
description: Iterative auto-research sub-agent that hill-climbs SAE features or circuit sites against a reward metric. Use when the discovery space is too large or context-heavy for the master agent to search inline — e.g. "find the top SAE features driving sycophancy / refusal / IOI on a heldout split, optimizing combined_auc_k."
---

# Circuit Discovery

Use the `discover_features` Tier-2 tool to spawn an inner agentic loop that
proposes, evaluates, and refines feature-scoring algorithms iteratively.
The sub-agent runs in its own session directory under
`<run_dir>/discovery/<session_name>/`, writes successive
`algorithm_v{N}.py` candidates, evaluates each via the local harness, and
keeps a leaderboard so it can hill-climb the chosen reward.

## When to use
- Master-agent context would blow up if it ran the search inline
  (50+ candidate evaluations, each with multi-page logs).
- The search needs concrete code-write/evaluate cycles — gradient-based
  attribution variants, alternate ablation strategies, position-specific
  scoring.
- You already have a reward metric registered (`combined_auc_k`, `auroc`,
  or a custom one declared in the spec).

## When NOT to use
- The reward can't be evaluated cheaply per candidate (the inner loop
  needs many evals; if each costs 10 GPU-min the loop won't converge).
- The work is a single-shot computation — just call `compute_metric`.
- You haven't pinned a reward yet. Discovery without a falsifiable target
  produces ranked nonsense.

## Workflow

1. **Pick the reward.** Default: `combined_auc_k` (= 0.5 *
   `mean_ablation_auc_k` + 0.5 * `mean_steering_auc_k`). If your phenomenon
   is a per-example ranking / classification question, prefer `auroc`.
   Read the metric card under `metrics/<name>.md` for input contracts.
2. **Confirm stage allowance.** `discover_features` is gate-restricted to
   stages whose `tools` list includes `discover_features`. Use
   `current_stage` to check; if missing, request a spec revision.
3. **Call `discover_features`** with a clear `task` and the reward name.
   Use `dry_run=true` first to confirm the harness round-trips before
   spending budget on real LLM calls.
4. **Inspect results.** The tool returns
   `{best_candidate, best_summary, session_dir, iterations_run, ...}`.
   `best_summary.top_features` is the ranked output ready to commit.
5. **Record evidence properly.** The discovery tool does NOT bypass the
   pre-registration gates. To attach the result to the run:
   - `commit_artifact("CandidateSite" | "FeatureFinding", payload, ...)`
     — the per-feature evidence.
   - `compute_metric(reward_metric, inputs=...)` on a heldout split, then
     `commit_artifact("MetricResult", payload, provenance_token=...)` —
     the reward value bound to a frozen split.
6. **Evaluate the criterion** if the spec ties one to discovery output.

## Sub-agent contract (for reference)

The sub-agent inside the session directory:
- writes one `algorithm_v{N}.py` per iteration starting from
  `algorithm_template.py`;
- runs `python <repo>/.../discovery/evaluate.py --session-dir . --algorithm
  ./algorithm_v{N}.py --candidate-name algorithm_v{N}` (with
  `--dry-run` when the harness should skip model loading);
- appends a JSON line to `experiments.jsonl` per candidate;
- updates `scratchpad.md` with `[NEXT] <plan>` or `[DONE] <summary>`;
- terminates when the LAST status marker is `[DONE]` or the iteration
  budget is exhausted.

The harness deterministic stub uses synthetic curves; real runs need a
project-specific evaluator passed via `--evaluator module:attr`.

## Cautions

- The sub-agent has only `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`
  inside its session dir. It cannot read other run artifacts directly,
  cannot call autointerp gates, and cannot edit files outside its session.
- Reward shopping is the obvious failure mode. Pre-register the reward in
  the spec — never let the sub-agent return mid-search and ask the master
  for a different reward.
- `combined_auc_k` weights ablation and steering equally. If your domain
  clearly favors one mode, declare a `CustomMetricDef` instead of
  rebalancing post-hoc.
- Evidence from discovery is descriptive until validated. Always re-evaluate
  the chosen circuit on a heldout split before claiming faithfulness.
