# autointerp

`autointerp` is an open-source scaffold for automated mechanistic interpretability agents.
It combines:

- a reusable library of interpretability tools in `src/autointerp/`;
- Codex-style method skills in `skills/`;
- an ML Intern-inspired agent runtime in `src/autointerp_agent/`;
- a Stage 0 conversational planner that produces pre-registered, falsifiable
  investigation specs (see [docs/stage0.md](docs/stage0.md));
- an investigation pipeline that executes approved specs through a coding
  agent under a gated tool surface (see
  [docs/investigation.md](docs/investigation.md));
- integration docs for downstream projects such as `circuitbreaker`.

The first target is practical agentic interpretability work: plan an investigation conversationally, pre-register a falsifiable spec, run local or MCP-style tools, use black-box probes when they are the cheapest useful method, cache activations before expensive white-box passes, and validate mechanistic claims causally.

For a step-by-step end-to-end walkthrough (install → spec → investigation →
results), see [QUICKSTART.md](QUICKSTART.md). For the IOI run report, see
[docs/IOI_END_TO_END_REPORT.md](docs/IOI_END_TO_END_REPORT.md).

## Install

```bash
git clone https://github.com/attribution-methods/autointerp.git
cd autointerp
python -m pip install -e ".[mcp]"
```

Install the heavier white-box analysis dependencies when you want local model
loading, activation hooks, probes, SAEs, or steering:

```bash
python -m pip install -e ".[mcp,mechinterp]"
```

Set an LLM API key for agent mode:

```bash
export ANTHROPIC_API_KEY=...
# or
export OPENAI_API_KEY=...
```

## Quick Start

List available skills:

```bash
autointerp --list-skills
```

Use a custom skill pack:

```bash
autointerp --skills-dir /path/to/skills --list-skills
```

Run a headless investigation:

```bash
autointerp "Plan an investigation of sycophancy using black-box probes first, then SAE features."
```

Run the CI-safe end-to-end scaffold example:

```bash
python examples/blackbox_to_validation.py
```

Start an interactive session:

```bash
autointerp
```

### Stage 0 — design an investigation conversationally

The agent enters Stage 0 spec-mode when you open with a research question.
It will recommend a model, walk through behavior framing → contrast →
dataset → hypothesis → stages → metrics, present the rendered spec, and
finalize on your approval. The approved spec is written to
`outputs/specs/<spec_id>_rev<n>.json`.

```bash
autointerp --model anthropic/claude-sonnet-4-6
> how does the model perform indirect object identification?
```

See [docs/stage0.md](docs/stage0.md) for the full design (closed
vocabularies, validation layers, approval flow, revision DAG).

### Investigation pipeline — execute an approved spec

Once a spec is approved, hand it to the investigation pipeline. The agent
walks `spec.stages` in order under a gated tool surface that mechanically
enforces pre-registration: the `success_criteria`, `metrics`, `abort_if`,
`dataset`, and `contrast` cannot be edited mid-run, metric values cannot be
forged, criteria can only be evaluated once, and split-tagged artifacts
block discovery/validation contamination.

```bash
python -m autointerp_agent.investigation \
  --spec outputs/specs/<spec_id>_rev<n>.json
```

Run directories land at `runs/<spec_id>_rev<n>/` with a frozen `spec.json`,
durable `state.json` (atomic writes, fully resumable across processes),
typed artifact subdirs, append-only `log.jsonl` audit, an
`INVESTIGATION_LOG.md` for the agent's narrative notes, and a final
`report.json`. The agent's escape hatch when it disagrees with the spec is
`request_spec_revision`, which terminates the run and points Stage 0 at a
child spec.

See [docs/investigation.md](docs/investigation.md) for the gate APIs
(`commit_artifact`, `compute_metric`, `evaluate_criterion`, `advance_stage`,
…), the run-directory layout, the `state.json` schema, and the provenance
token mechanics that close the "agent fabricates a number" hole.

#### Watching a run / debugging after the fact

Every run captures a complete transcript on disk:

- `assistant_turns.jsonl` — every LLM assistant message (full text +
  tool-call summaries).
- `tool_invocations.jsonl` — every tool call (args, ok, output preview).
- `tool_invocations/<iter>_<idx>_<tool>_<id>.txt` — full untruncated tool
  output (so bash stdout/stderr, raw JSON, error traces are all replayable).
- `log.jsonl` — append-only audit of *gated* tool calls (`compute_metric`,
  `commit_artifact`, `evaluate_criterion`, `advance_stage`, abort/budget
  trips) with structured args and budget snapshots.
- `state.json` — atomic-write bookkeeping (current stage, criteria,
  budget, terminal state). Always consistent.
- `INVESTIGATION_LOG.md` — the agent's own narrative notes.
- `scripts/`, `scratch/` — every Python file the agent wrote and any
  intermediate dumps.

Live, you'll see one line per assistant turn and one line per tool call on
the console (use `--quiet` to suppress; the disk transcript is unaffected).
Tail-follow `assistant_turns.jsonl` and `tool_invocations.jsonl` from
another terminal for live structured inspection.

## Repository Layout

```text
src/autointerp/                       Mechanistic interpretability methods and helpers
src/autointerp/schemas.py             Shared investigation artifact schemas
src/autointerp/spec.py                Stage 0 InvestigationSpec + closed vocabularies
src/autointerp/spec_partial.py        PartialSpec used during conversational drafting
src/autointerp/spec_describe.py       Schema reference rendered for the agent
src/autointerp/pipelines/investigation/  Investigation pipeline gates (run_dir, state,
                                          artifacts, metrics, criteria, guards, stages,
                                          revision, report, tools, main)
src/autointerp_agent/                 Agent runtime, CLI, tools, context, permissions
src/autointerp_agent/stage0_tools.py  Stage 0 tools: describe_spec, update_spec, etc.
src/autointerp_agent/investigation.py CLI entry point for the investigation pipeline
metrics/                    Metric reference cards (one .md per MetricName)
examples/                   Runnable scaffold examples
skills/                     Reusable method skills
configs/agent.yaml          Default agent configuration
docs/                       Architecture, schemas, Stage 0 notes
integrations/               Downstream integration notes
```

## Scaffold Attribution

The agent runtime is intentionally modeled on the public Hugging Face `ml-intern` scaffold, especially its CLI/headless modes, tool router, planning tool, permission checks, local/sandbox tool split, and MCP support. See [NOTICE.md](NOTICE.md) and [docs/ml_intern_scaffold_notes.md](docs/ml_intern_scaffold_notes.md).

## Development

Validate skills:

```bash
python scripts/validate_skills.py
```

Run the local test suite:

```bash
pytest -q
```

Smoke-test imports:

```bash
python -B -c "from autointerp_agent import load_config; from autointerp_agent.skills import SkillRegistry; print(len(SkillRegistry.from_repo().skills))"
```

See [docs/schemas.md](docs/schemas.md) for the shared artifact contracts,
[docs/stage0.md](docs/stage0.md) for the Stage 0 conversational planner,
[docs/end_to_end_mvp.md](docs/end_to_end_mvp.md) for the initial investigation
path, and [docs/discovery_subagent.md](docs/discovery_subagent.md) for the
iterative `feature_discovery` sub-agent (Tier-2 `discover_features` tool).
