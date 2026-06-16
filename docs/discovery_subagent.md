# Discovery Sub-Agent (hill-climbing)

A small, provider-agnostic hill-climbing search that an investigation can
delegate to when the bottleneck is *which algorithm finds the causal sites*,
not a single concrete site. Exposed as the Tier-2 `discover_features` tool.

## Engine

`src/autointerp/pipelines/investigation/discovery/` is a generic engine
(`run_hillclimb`) plus a thin feature-discovery wrapper (`run_discovery_subagent`):

| File | Role |
|---|---|
| `task.py` | `HillClimbTask` (what to optimize), `HillClimbConfig` (how hard), `HillClimbResult`. |
| `engine.py` | `run_hillclimb` — archive (population), parallel width, memory, budget, seed-variance, resume. |
| `proposer.py` | `propose_single` (one LLM completion) and `propose_agentic` (tool-using subagent via `run_agent_turn`). |
| `harness.py` | Evaluate ONE candidate → reward JSON. `--dry-run` stub + `--evaluator module:attr` plug. |
| `algorithm_template.py` / `candidate.py` | The `score(...) -> list[Candidate]` contract + deterministic baseline. |
| `system_prompt.py` | Task-driven prompt scaffold (contract injected per task). |
| `loop.py` | Feature-discovery instantiation of `run_hillclimb`. |

The engine is task-agnostic: adding a new optimization target (prompt search,
probe search, ...) means building a different `HillClimbTask` and calling
`run_hillclimb` — no engine change.

## Loop

Each round: pick parents from the archive (best + diverse) → propose
`n_subagents` candidates concurrently → evaluate each via the harness → merge
into a top-K archive → stop on no-improvement (`patience`), perfect reward,
exhausted token budget, or `max_iterations`. The best candidate is then
re-evaluated across `seeds` to report reward variance (fluke screening).

Default reward: `combined_auc_k = 0.5*(mean_ablation_auc_k + mean_steering_auc_k)`.

## Provider-agnostic, no extra SDK

Proposals reuse the runtime's LiteLLM path (`_call_llm` / `run_agent_turn`), so
any provider LiteLLM supports works (Anthropic, OpenAI, OpenRouter, local
vLLM/HF) by setting `DiscoveryConfig.model`. No `claude-agent-sdk` dependency.

## Memory

Bounded context + structured lookup (avoids re-sending every program each
proposal):

- **In the prompt (cheap, fixed-size):** a deterministic progress summary
  (best reward, attempts-since-best, failure counts), a compact leaderboard
  (one line per archived candidate, no code), the recent trail (last ~12
  attempts; older ones noted as a count), and FULL code for only the parent +
  `archive_code_in_context` top candidates (default 1).
- **Structured memory on disk:** `archive.jsonl` (population, enables resume),
  `results/<candidate>.json`, each `<candidate>.py`, `loop_log.jsonl`,
  `cost.json`.
- **Agentic mode** points the subagent at the on-disk store and lets it read
  any candidate's full code with its file tools — full history is available on
  demand without inflating every prompt. Single-completion mode can't look up,
  so it relies on the bounded inline context (raise `archive_code_in_context`
  if it needs more).

## Pre-registration discipline

`discover_features` is **advisory** — it commits no gated artifacts. The master
agent records evidence the normal way: `commit_artifact` for
`CandidateSite`/`FeatureFinding`, and `compute_metric` + `commit_artifact` for
the reward on a **heldout** split. It is stage-gated (must be in the stage's
`tools`) and its reward must be one of the stage's pre-registered `metrics`.

## Hyperparameters (frozen in the spec)

`DiscoveryConfig` on a `StageSpec` holds the reproducibility-relevant knobs:
`max_iterations`, `patience`, `n_subagents` (width), `archive_size` (population),
`propose_mode` (`single`|`agentic`), `max_turns_per_iteration`, `model`,
`max_tokens`, `top_k`, `k_grid`, `objective`, `seeds`, `evaluator`. Authored in
Stage 0 (surfaced by `describe_spec`). Ephemeral switches (`dry_run`) stay as
tool args.

## Budget

Proposal token/USD spend is accumulated via `CostTracker` and bridged into
`state.budget_consumed.tokens` / `.cost_usd` by the tool handler. The run's
`spec.budget.max_tokens` is enforced (newly wired), and the discovery call is
capped at the remaining token budget so the inner loop can't overrun the run.

## Dry-run

`dry_run=True` skips the LLM: one deterministic candidate from the template,
evaluated by the harness stub. Used by tests/CI (`tests/test_discovery_loop.py`)
to round-trip the whole loop, gating, budget, and report without GPUs or creds.

## Related work

The design tracks the "autoresearch" family: a single-branch hill climb
(Karpathy's AutoResearch) extended with a population/archive and parallel width
(AlphaEvolve, ScaleAutoResearch), persistent trail memory, and multi-seed
fluke screening (Recursive). The pre-registration gates are the analog of their
"increasingly strict validation / immutable verifier."
