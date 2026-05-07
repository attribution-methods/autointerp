# Discovery Sub-Agent — Design Notes

This document is the implementation note for the
`feature_discovery` stage and the `discover_features` Tier-2 tool added on
the `discovery-subagent` branch (Phase A + Phase B). It mirrors the design
of [circuitbreaker/auto_circuit_discovery](https://github.com/attribution-methods/circuitbreaker/tree/auto-circuit-discovery/auto_circuit_discovery)
adapted to autointerp's run-handle, gates, and metric registry.

## Why a sub-agent

`discover_features` is invoked when the iterative search space (proposing,
evaluating, refining ranking algorithms or attribution methods) would
otherwise blow up the master agent's context. The master delegates the
inner search loop to a fresh Claude Code SDK sub-agent that:

- runs in its own session directory under `<run_dir>/discovery/<name>/`;
- writes successive `algorithm_v{N}.py` candidates from a template;
- evaluates each via a stable harness CLI;
- maintains a leaderboard + memory summary that the next iteration's
  prompt reads;
- terminates by writing `[DONE]` to its `scratchpad.md`.

The master receives one structured object (best candidate + top features
+ session path) and proceeds with the *gated* recording path
(`commit_artifact` / `compute_metric`).

## Architecture

```
        Master (outer) agent — run_agent_turn(...)
               │  Tier-2 tool: discover_features
               ▼
    pipelines.investigation.discovery.run_discovery_subagent
               │
   ┌───────────┴────────────────────────────────────────────┐
   │   Per-iteration (max_iterations):                       │
   │     - build prompt (scratchpad + memory + leaderboard)  │
   │     - claude_code_sdk.query(...) → sub-agent            │
   │         allowed_tools: Read/Write/Edit/Bash/Glob/Grep   │
   │         cwd:           session_dir                      │
   │     - sub-agent runs evaluate.py → harness.py           │
   │     - session.update_session_files(...) writes          │
   │       leaderboard.md / memory_summary.md                │
   │   loop until [DONE] or max_iterations / timeout         │
   └─────────────────────────────────────────────────────────┘
```

### Files (`src/autointerp/pipelines/investigation/discovery/`)

| File | Role |
|---|---|
| `loop.py` | Outer iterative loop (`run_discovery_subagent`), `[DONE]` detection, dry-run path. |
| `session.py` | Deterministic `leaderboard.md` and `memory_summary.md` writers from `results/*/summary.json`. |
| `evaluate.py` | Bridge: resolves the candidate path, spawns harness as a subprocess, writes `summary.json`. |
| `harness.py` | Loads candidate, calls `score(...)`, computes AUC_K via the autointerp metric registry. Has a `--dry-run` stub for CI. |
| `algorithm_template.py` | The `score(...)` contract + a deterministic baseline candidate. |
| `system_prompt.py` | Discovery sub-agent system prompt template (reward-aware). |

### Auth

The sub-agent uses `claude_code_sdk.query(...)`, which spawns the local
`claude` CLI. The CLI uses subscription OAuth from `~/.claude/.credentials.json`,
so users on Claude Pro / Max / Team need *no* `ANTHROPIC_API_KEY`. The
*outer* agent is unchanged — it still uses LiteLLM, which can route through
either an API key or OpenRouter.

The two model configurations are independent:
- **Outer agent**: `configs/agent.yaml` `model_name`, default
  `anthropic/claude-sonnet-4-5` via LiteLLM.
- **Sub-agent**: `discover_features` `model` arg, default
  `claude-sonnet-4-5` via `claude_code_sdk`.

## What's gated and what isn't

`discover_features` is registered on the Tier-2 tool surface but only
*reads* the spec/state — it does not commit artifacts, evaluate criteria,
or advance stages. Its outputs are advisory:

- The master must call `commit_artifact("CandidateSite", ...)` or
  `commit_artifact("FeatureFinding", ...)` to record per-feature
  evidence.
- The master must call `compute_metric(reward_metric, inputs=...)` on a
  heldout split, then `commit_artifact("MetricResult", ...)` with the
  returned provenance token, to record the reward as a checkable value.

This preserves all existing pre-registration invariants. The sub-agent
*cannot fabricate a metric value* — only the master can mint provenance
tokens, and only via `compute_metric`.

`discover_features` *is* gate-restricted by stage: the handler refuses
unless the active stage's `tools` list includes `ToolName.DISCOVER_FEATURES`.

## Dry-run mode

`run_discovery_subagent(..., dry_run=True)` skips the SDK entirely:

- A single deterministic iteration copies `algorithm_template.py` to
  `algorithm_v1.py`.
- The harness runs in `--dry-run` mode (no model load) and synthesizes
  per-pair AUC curves from the candidate's returned magnitudes.
- The scratchpad is written with `[DONE]` after one iteration so the
  outer loop terminates immediately.

Used by `tests/test_discovery_loop.py` and by the CLI to confirm the
plumbing without GPUs or subscription auth.

## Live run instructions

### Prerequisites

1. **Install the SDK** (added to `pyproject.toml` only as a soft optional —
   the import is deferred):
   ```bash
   pip install claude-code-sdk
   ```
2. **Log into the Claude CLI with your subscription**:
   ```bash
   claude /login
   ```
   On a fresh machine this opens a browser to your Claude.ai account and
   stores OAuth tokens under `~/.claude/`.
3. **Install autointerp + heavier deps** if you haven't:
   ```bash
   pip install -e ".[mcp,mechinterp]"
   ```

### One-shot smoke (no LLM, no GPU)

```bash
PYTHONPATH=src python -m pytest tests/test_discovery_loop.py -q
PYTHONPATH=src python -m autointerp_agent.app validate \
  --spec examples/specs/feature-discovery-demo-v1_rev1.json
```

### Live discovery from inside an investigation

Once the demo spec is approved (it's already `status=approved`), launch
the investigation pipeline; it will execute the `feature_discovery` stage
with the sub-agent live:

```bash
PYTHONPATH=src python -m autointerp_agent.investigation \
  --spec examples/specs/feature-discovery-demo-v1_rev1.json \
  --auto-approve
```

The master agent will (a) run the black-box stage on GPT-2-small, (b)
call `discover_features` in the discovery stage, (c) record the
top-feature evidence and the heldout reward, (d) terminate with the
criterion check. The sub-agent's session lives at
`runs/feature-discovery-demo-v1_rev1/discovery/<session_name>/` for
inspection.

### Plugging in a real evaluator

The `--dry-run` harness uses synthetic curves. For a real run, write an
evaluator in your own module:

```python
# my_project/evaluator.py
def evaluate(score_fn, *, top_k, k_grid, layers, device, context):
    # Load model + SAEs once.
    # For each (clean, corrupted) pair:
    #   call score_fn(...) -> ranked candidates
    #   ablate top-K, measure normalized delta over k_grid
    #   steer top-K, measure normalized delta over k_grid
    # Return:
    return {
        "mean_ablation_auc_k": <float in [0,1]>,
        "mean_steering_auc_k": <float in [0,1]>,
        "top_features":         [{"layer": ..., "feature_id": ..., "score": ...}],
    }
```

Then point the harness at it via `--evaluator my_project.evaluator:evaluate`
when the sub-agent runs `evaluate.py`. The simplest place to do that is
to add a thin wrapper on top of `evaluate.py` that hard-codes
`--evaluator` for your project, and tell the sub-agent to call the
wrapper instead via the per-iteration prompt.

## Testing

| What | Command |
|---|---|
| AUC metric impls | `PYTHONPATH=src pytest tests/test_auc_metrics.py -q` |
| Session writers + `[DONE]` | `PYTHONPATH=src pytest tests/test_discovery_session.py -q` |
| End-to-end dry-run | `PYTHONPATH=src pytest tests/test_discovery_loop.py -q` |
| Tier-2 tool registration | `PYTHONPATH=src pytest tests/test_investigation_entrypoint.py -q` |
| Spec parses with new enums | `PYTHONPATH=src python -m autointerp_agent.app validate --spec examples/specs/feature-discovery-demo-v1_rev1.json` |

## Known follow-ups

- Real GPU-backed evaluator (current harness only ships the deterministic
  stub — wire a Gemma/Llama + SAE evaluator next).
- Cost / budget bridging: the sub-agent's `cost_usd` is not yet
  contributed to the parent run's `state.budget_consumed`. Plumb a
  budget hook so the run aborts cleanly when the inner loop exhausts
  GPU-seconds.
- Resume semantics: `init_session` is idempotent on the session dir but
  `run_discovery_subagent` does not currently restart from the last
  iteration in `loop_log.jsonl`. Mirror the reference's resume path
  next.
- Skill fattening: existing `skills/sparse-autoencoders/SKILL.md` and
  `skills/circuit-tracing/SKILL.md` are still boilerplate. Per the issue
  #6 scope, fold the "Algorithm Ideas to Try" content into them so the
  sub-agent (and master) inherit concrete recipes.
