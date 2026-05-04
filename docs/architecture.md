# Architecture

`autointerp` has three layers:

1. `autointerp`: schemas, pipelines, and domain tools for interpretability.
2. `skills`: procedural knowledge for choosing and applying methods.
3. `autointerp_agent`: an agent runtime that can select skills, call tools, request approval, and run locally or through MCP-style integrations.

## Runtime Loop

```text
user prompt
  -> ContextManager builds system + skill context
  -> LiteLLM model call with ToolRouter specs
  -> zero or more tool calls
  -> permission check
  -> tool execution
  -> tool output appended to context
  -> [optional] observer hooks fire (assistant_turns.jsonl,
     tool_invocations.jsonl + bodies, console stream)
  -> repeat until final answer or max iterations
```

`run_agent_turn` accepts an optional ``observer`` (a ``TurnObserver``
protocol). The investigation CLI attaches a ``RunObserver`` automatically;
ad-hoc tooling can pass any object that implements
``on_iteration_start`` / ``on_assistant`` / ``on_tool_call`` / ``on_final``
hooks. Observer exceptions are swallowed so a broken logger cannot crash
the run.

## Tool Policy

The initial runtime includes local tools because the near-term bottleneck is engineering:

- `plan`: maintain a visible task plan.
- `list_skills`: inspect available method skills.
- `read_skill`: load a skill's `SKILL.md`.
- `bash`: run local commands with timeout and output truncation.
- `read_file`: read files with line numbers.
- `write_file`: write files after read-before-write checks.
- `edit_file`: exact string replacement after read-before-write checks.
- Stage 0 tools (`describe_spec`, `update_spec`, `remove_spec_fields`,
  `show_spec`, `validate_spec`, `finalize_spec`, `list_metrics`,
  `read_metric`): build a pre-registered investigation spec
  conversationally. See [stage0.md](stage0.md).
- MCP tools: optional, loaded from config when `fastmcp` is installed.

Potentially destructive or expensive operations require approval unless `auto_approve` is set.

## Domain Layer

The method library includes:

- black-box audit probes;
- model loading and chat formatting;
- activation extraction and component specs;
- contrastive directions;
- activation steering and patching;
- logit lens and direct logit attribution;
- probes and gradient attribution;
- SAE feature post-processing;
- circuit ranking helpers.

The skills describe when and how to use these methods. The code implements repeatable operations.

## Artifact Layer

`autointerp.schemas` defines the portable JSON contracts that let components
exchange evidence:

- `BehaviorSpec`, `PromptBatch`, and `GenerationSample` for black-box work;
- `ActivationCacheRef`, `CandidateSite`, and `FeatureFinding` for white-box
  discovery;
- `InterventionResult`, `ValidationResult`, and `InvestigationReport` for
  causal checks and summaries.

Large tensors stay out of report JSON and are referenced by path. This keeps the
same report usable by the CLI agent, circuitbreaker, notebooks, and later MCP
servers.

## Investigation Pipeline (executes an approved spec)

`src/autointerp/pipelines/investigation/` is the runtime that takes an
approved `InvestigationSpec` and executes it through a coding agent under a
gated tool surface. Same agent shape as Stage 0 (LiteLLM + `ToolRouter`),
different system prompt and a small Tier-2 tool set on top of the Tier-1
free-form tools.

Key invariants enforced in code, not by trust:

- `spec.json` is `chmod r--` after `init_run`. The only path to deviate
  from a frozen spec is `request_spec_revision` (Stage 0 then drafts a
  child).
- `compute_metric` is the only legitimate way to produce a `MetricResult`
  value. It issues a one-time provenance token; `commit_artifact` for a
  `MetricResult` requires the token and verifies value/metric_id/stage_idx
  match.
- `evaluate_criterion` runs once per criterion per spec rev. Failed
  criteria flip the run terminal as `CRITERION_FAILED`.
- Every committed artifact is split-tagged via
  `metadata["_provenance"]`; criteria with `on_split="heldout"` refuse
  inputs not tagged heldout.
- Typed `AbortPredicate` entries in `spec.abort_if` are mechanically
  evaluated after every metric commit and trip the run to `ABORTED`.
- `spec.budget` is enforced at every gated entry; exhaustion sets
  `terminal_state = BUDGET_EXHAUSTED`.

CLI: `python -m autointerp_agent.investigation --spec <path>`. Run
directories live at `runs/<spec_id>_rev<n>/`. State is durable via atomic
`state.json` writes; runs resume cleanly across processes.

Observability is built in. Each run captures a complete transcript:

- `assistant_turns.jsonl` — every LLM assistant message (full content +
  tool-call summaries).
- `tool_invocations.jsonl` — every tool call (args, ok, output preview,
  pointer to full body).
- `tool_invocations/<iter>_<idx>_<tool>_<id>.txt` — full untruncated tool
  output for replay (bash stdout/stderr, raw JSON, error traces).
- `log.jsonl` — append-only audit of *gated* calls only (`compute_metric`,
  `commit_artifact`, `evaluate_criterion`, `advance_stage`, abort/budget
  trips) with structured args and budget snapshots.
- `INVESTIGATION_LOG.md`, `scripts/`, `scratch/` — the agent's narrative
  notes, code it wrote, and intermediate dumps.

Live, the CLI also prints one line per assistant turn and tool call to the
console (use `--quiet` to suppress; disk transcript is unaffected).

See [investigation.md](investigation.md) for the full design, run-dir
layout, state schema, gate APIs, observer hooks, and provenance token
mechanics.

## Stage 0 — Investigation Specification

Before any expensive run, the agent and the user co-design a pre-registered
`InvestigationSpec` (`autointerp.spec`). The conversation is driven by a
small set of Stage 0 tools that read from a typed schema (`describe_spec`),
closed-vocabulary metrics (`list_metrics` / `read_metric`), and a draft
that's auto-saved to `outputs/specs/_draft.json` between turns.

Validation is layered:

1. **Prevention at write time** — `update_spec` rejects unknown top-level
   keys with field-name suggestions.
2. **Per-field schema (pydantic)** — types, enums, metric range/direction
   contracts, tool→required-field deps, spec-level invariants.
3. **Cross-field validator** — `validate_spec` wraps `try_build` and
   pretty-prints errors so the agent self-checks mid-conversation.

Methodological judgment (sample size, dataset choice, thresholds) is the
user's responsibility. Code catches only what code can prove wrong; the user
reviews methodology when the rendered spec is presented for approval.

`finalize_spec` is two-phase: phase 1 renders the spec for the user, phase 2
writes the approved spec to `outputs/specs/<spec_id>_rev<n>.json` with an
`Approval` record. Approved specs are immutable on the current revision —
revisions form a DAG (`parent_spec_id`, `prior_results_ref`).

See [stage0.md](stage0.md) for the full specification, tool surface, and
conversational design.

## Near-Term Direction

The team should avoid full all-layer/all-token SAE decoding by default. Use cheap localization first:

1. black-box probes;
2. ablations or activation patching to find causal layers/sites;
3. targeted SAE feature inspection;
4. causal validation on held-out prompts.
