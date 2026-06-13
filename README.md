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
python -m pip install -e .
```

That's everything needed for spec design, API/black-box investigations, and
inspecting runs. One optional extra exists, `mechinterp`, for **local
white-box model work** (loading models, activation hooks, patching, probes,
SAEs, steering). It's opt-in because it pulls in torch — several GB of CUDA
wheels — which you don't want on laptops, in CI, or on GPU boxes that
already ship a system torch matched to their driver:

```bash
python -m pip install -e ".[mechinterp]"   # add local/white-box model support
```

Set an LLM API key for agent mode:

```bash
export ANTHROPIC_API_KEY=...
# or
export OPENAI_API_KEY=...
# or
export OPENROUTER_API_KEY=...
```

No key exported? Just run `autointerp`: when the configured model's provider
has no key in the environment, the CLI opens an interactive setup — pick a
provider (arrow keys), paste the key (masked), then pick from a **model list
fetched live from the provider's `/models` endpoint** (so it always shows
what your key can actually call; offline it falls back to litellm's bundled
registry, then a minimal built-in list). The agent sends tool definitions on
every call, so the OpenRouter list is filtered to tool-capable models and
the validation ping exercises tool calling — models that can't drive the
agent fail at setup, not mid-session. The choice is validated, then offered
for saving (key + `AUTOINTERP_MODEL`) to either
`~/.autointerp/credentials` (recommended — applies in any directory, file
mode 0600) or a project-local gitignored `.env`. Precedence follows the
usual CLI convention:

```text
process env (export …)  >  ./.env (project)  >  ~/.autointerp/credentials (user)
```

Inside any interactive session, `/model` re-opens the picker to switch
model, provider, or key mid-conversation — switching to a provider whose key
isn't set (e.g. OpenAI → Anthropic) prompts for that key right there; if a
key already exists you can keep it or replace it (`/help` lists all
commands). `--model` still overrides per invocation, and a free-text
"custom" entry accepts any litellm model string.

**Local / open-weights models.** The picker also has a **Local
(HuggingFace)** option: pick a curated open-weights chat model (Qwen2.5 /
Qwen3 / Llama-3.x, 7B–72B, all sized to fit a ≤180 GB GPU in bf16) or type
any HuggingFace id. No API key — it routes through litellm's `hosted_vllm/`
to a local OpenAI-compatible server (default `http://localhost:8000/v1`,
configurable). Serve the model with tool-calling enabled, e.g.:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
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

### Use these skills in Codex or Claude Code

The folders under `skills/` are Agent Skills: each skill is a directory with a
`SKILL.md` file plus optional metadata/resources. They can be used outside the
`autointerp` CLI.

The plugin path is the easiest way to install the full skill bundle because it
keeps the skills versioned and updateable through the agent's plugin manager.

Install the skill bundle as a Codex plugin:

```bash
codex plugin marketplace add attribution-methods/autointerp \
  --sparse .agents/plugins \
  --sparse plugins/autointerp-skills
codex plugin add autointerp-skills@autointerp
```

Install the skill bundle as a Claude Code plugin:

```bash
claude plugin marketplace add attribution-methods/autointerp \
  --sparse .claude-plugin plugins/autointerp-skills
claude plugin install autointerp-skills@autointerp
```

To test a local checkout before installing from GitHub:

```bash
codex plugin marketplace add ./
codex plugin add autointerp-skills@autointerp

claude plugin marketplace add ./
claude plugin install autointerp-skills@autointerp
```

For direct local copying instead of plugin management, install all skills for
Codex:

```bash
mkdir -p ~/.agents/skills
cp -R skills/* ~/.agents/skills/
```

Install all skills for Claude Code:

```bash
mkdir -p ~/.claude/skills
cp -R skills/* ~/.claude/skills/
```

For project-local use instead of personal/global use, copy to
`.agents/skills/` for Codex or `.claude/skills/` for Claude Code.

To download only the raw skill folders from GitHub:

```bash
tmp="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/attribution-methods/autointerp.git "$tmp/autointerp"
git -C "$tmp/autointerp" sparse-checkout set skills
mkdir -p ~/.agents/skills ~/.claude/skills
cp -R "$tmp/autointerp/skills/"* ~/.agents/skills/
cp -R "$tmp/autointerp/skills/"* ~/.claude/skills/
rm -rf "$tmp"
```

To install only one skill, copy that subdirectory instead, for example
`skills/relevance-patching`. Review third-party skills before installing them;
Codex and Claude Code may let skills include supporting scripts/resources.
If a newly copied skill does not appear in an already running agent session,
restart the agent.

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

The interactive shell opens with a welcome banner (model, key status,
skills, version) and behaves like a modern agent CLI: a bordered input with
slash-command autocompletion (type `/`), cross-session history and a status
bar with live token/cost totals, tool-call indicators with a spinner while
the agent works, markdown-rendered answers, and Ctrl-C interrupting the
current turn instead of the session. First launch runs a one-time setup
(provider → key → live model list) persisted to `.env` + `~/.autointerp/`;
after that it never re-runs.

Session commands (autocomplete after typing `/`):

- `/model` — switch model / provider / API key mid-session (live model
  list from the provider, validated before switching)
- `/litrev [topic]` — side-channel **arXiv literature review**: a panel of
  linked papers with 3–4 sentence summaries, also saved as markdown to
  `outputs/litrev/`. The topic defaults to the draft spec's question (or
  your last message). Results never enter the agent's context — mention a
  paper yourself if you want the agent to use it.
- `/cost` — per-model token and dollar breakdown for the session
- `/status` — model, key, turns, context size, draft-spec state
- `/skills`, `/clear`, `/help`, `/exit`

`autointerp` always starts a **fresh** session (a stray draft plan from an
interrupted session is archived, never silently resumed). Resume explicitly:

```bash
autointerp --continue    # arrow-key picker over this project's sessions
```

Every session is saved per project under `~/.autointerp/sessions/` after
each turn — conversation, draft plan, turn count, and cost. Resuming
restores the agent's full context and shows a "session resumed" panel;
`/clear` rotates to a new session in place (the old one stays resumable).

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

Validation includes model/method coherence: a spec whose stages declare
white-box tools (lenses, activation patching, SAEs, probes, steering, …)
is mechanically rejected if it targets a known closed-weights API model
(GPT-4/5, o-series, Claude, Gemini, …) — those support black-box stages
only, since every other tool loads the model's weights locally.

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
[docs/stage0.md](docs/stage0.md) for the Stage 0 conversational planner, and
[docs/end_to_end_mvp.md](docs/end_to_end_mvp.md) for the initial investigation
path.
