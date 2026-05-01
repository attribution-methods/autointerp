# autointerp

`autointerp` is an open-source scaffold for automated mechanistic interpretability agents.
It combines:

- a reusable library of interpretability tools in `src/autointerp/`;
- Codex-style method skills in `skills/`;
- an ML Intern-inspired agent runtime in `src/autointerp_agent/`;
- integration docs for downstream projects such as `circuitbreaker`.

The first target is practical agentic interpretability work: plan an investigation, run local or MCP-style tools, use black-box probes when they are the cheapest useful method, cache activations before expensive white-box passes, and validate mechanistic claims causally.

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

Validate the golden case-study catalog:

```bash
python scripts/validate_case_studies.py
```

Start an interactive session:

```bash
autointerp
```

## Repository Layout

```text
src/autointerp/          Mechanistic interpretability methods and helpers
src/autointerp/schemas.py Shared investigation artifact schemas
src/autointerp_agent/    Agent runtime, CLI, tools, context, permissions
examples/                Runnable scaffold examples
case_studies/            Public-safe golden case-study catalog
skills/                  Reusable method skills
configs/agent.yaml       Default agent configuration
docs/                    Architecture and scaffold notes
integrations/            Downstream integration notes
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

See [docs/schemas.md](docs/schemas.md) for the shared artifact contracts and
[docs/end_to_end_mvp.md](docs/end_to_end_mvp.md) for the initial investigation
path.
See [docs/case_studies.md](docs/case_studies.md) for the golden case catalog
and private ground-truth handling.
