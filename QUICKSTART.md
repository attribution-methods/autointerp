# Quickstart — Spec to Investigation, End-to-End

This walks through the full path: install → design a spec interactively →
run the investigation → read the results. The worked example is the IOI
run from `docs/IOI_END_TO_END_REPORT.md`, but the same flow works for any
behavior.

## 0. Install

```bash
git clone https://github.com/attribution-methods/autointerp.git
cd autointerp
python -m pip install -e ".[mcp,mechinterp]"
export ANTHROPIC_API_KEY=...
```

You'll need a GPU for any run that loads a real model (GPT-2-small needs
<2 GB, anything bigger needs a real card).

## 1. Design a spec (Stage 0) and run it — in one command

```bash
autointerp investigate "How does GPT-2-small mechanistically implement indirect object identification?"
```

`investigate` runs Stage 0 conversationally; the moment you approve a spec
via `finalize_spec`, it auto-launches the investigation pipeline against
that spec with sane defaults (`--max-iterations 500`, `--auto-approve`).
You can also stay one-shot:

```bash
autointerp investigate --spec examples/specs/ioi-gpt2-small-blind-v1_rev1.json
```

Or design first and run later:

```bash
autointerp                      # interactive Stage 0 REPL
# ...approve a spec, then later:
autointerp investigate --spec outputs/specs/<spec_id>_rev<n>.json
```

The agent enters spec-mode and walks you through:

1. **Behavior framing** — what the model does, what counts as success.
2. **Contrast** — clean vs. corrupt prompt template (e.g. ABC name swap
   for IOI). The agent enforces tokenization and length constraints.
3. **Dataset** — generator, sample size, dev/heldout split, seeds.
4. **Hypothesis** — your prior on the mechanism (kept honest: the agent
   is told to *discover*, not assume).
5. **Stages** — typically `black_box → localization → activation_analysis
   → intervention → validation`. Each stage names the tools and metrics
   it's allowed to use.
6. **Success criteria** — pre-registered, wired to the metric registry
   (see `metrics/*.md` for the full set: `accuracy`, `logit_diff`,
   `patch_effect_recovery`, `kl_to_clean`, `faithfulness`, `minimality`,
   `necessity_drop`, …).

When you approve the rendered spec, it's written to
`outputs/specs/<spec_id>_rev<n>.json` and **frozen** for the rest of the
run. After approval the agent cannot edit success criteria, thresholds,
metrics, or the dataset/contrast — only a new revision (with a human in
the loop) can change those.

A worked spec — exactly the one Sonnet ran for IOI — lives at
[examples/specs/ioi-gpt2-small-blind-v1_rev1.json](examples/specs/ioi-gpt2-small-blind-v1_rev1.json).
You can copy it into `outputs/specs/` and skip Stage 0 to reproduce the run.

## 2. Pipeline mechanics (what `investigate` does under the hood)

Output lands in `runs/<spec_id>_rev<n>/`. The agent walks `spec.stages`
in order; it can:

- write Python via `bash` / `write_file` / `edit_file` (Tier 1, free-form);
- call gated Tier-2 tools `compute_metric`, `commit_artifact`,
  `evaluate_criterion`, `advance_stage`, `request_spec_revision`,
  `current_stage`, `get_state`, `get_budget`.

Tier-2 is the only path that writes to `findings/` or moves the run
forward. Metric values come from registered functions over hashed
inputs — the agent cannot fabricate a number.

Useful flags on `autointerp investigate`:

- `--max-iterations N` — default **500** (was 60; too low killed our first run).
- `--auto-approve / --no-auto-approve` — auto-approve is **on** by default
  for non-interactive runs; flip it off if you want to confirm each tool.
- `--no-resume` — force a fresh run dir (default is *resume if partial*).
- `--quiet` — suppress live console streaming. On-disk transcript is unaffected.

Pre-flight check before a paid run:

```bash
autointerp validate --spec examples/specs/ioi-gpt2-small-blind-v1_rev1.json
# add --load-model to also confirm the model loads
```

## 3. Watch / inspect runs

```bash
autointerp runs list             # table: stage, status, criteria, wallclock
autointerp runs show <run_id>    # pretty-printed report.json
autointerp runs tail <run_id>    # live stream of tool calls + assistant turns
```

`runs tail` reads `tool_invocations.jsonl` and `assistant_turns.jsonl`
under the hood. For full untruncated tool output (bash stdout/stderr,
JSON dumps, error traces), open
`runs/<run_id>/tool_invocations/<iter>_<idx>_<tool>_<id>.txt`.

## 4. Read the results

When the run finishes (or hits `request_spec_revision`), look at:

- `runs/<spec>/report.json` — the final pre-registered evaluation. Every
  success criterion shows `passed`, `value`, `metric`, `comparator`,
  `threshold`, and a `metric_result_ref` pointing at the on-disk
  computation.
- `runs/<spec>/findings/stage_*/*.json` — committed metric results, each
  with `inputs_hash` (sha256 of the raw inputs the metric was computed
  on), `registry_version`, and `_provenance` (stage, split, timestamp).
- `runs/<spec>/scripts/*.py` — every Python script the agent wrote. They
  are runnable; you can re-execute any of them to verify the numbers
  reproduce. The IOI heldout validation script reproduces 0.913
  faithfulness exactly.

## 5. Reproduce the IOI run

```bash
autointerp validate --spec examples/specs/ioi-gpt2-small-blind-v1_rev1.json
autointerp investigate --spec examples/specs/ioi-gpt2-small-blind-v1_rev1.json
autointerp runs show ioi-gpt2-small-blind-v1_rev1
```

Expected outcome: discovers the circuit `[(L9, H9), (L9, H6), (L10, H0)]`,
all four pre-registered criteria pass, ~30 min wallclock on a single GPU.
See [`docs/IOI_END_TO_END_REPORT.md`](docs/IOI_END_TO_END_REPORT.md) for
the full write-up.

## 6. Add a new behavior

Just open `autointerp` and ask a different question; Stage 0 will draft a
fresh spec. To extend the toolkit while you're at it:

- New **metric**: drop a `metrics/<name>.md` card and register the
  implementation in the metric registry (`src/autointerp/metrics/`).
- New **skill**: drop `skills/<name>/SKILL.md`. Add tool entries to
  `src/autointerp/tools/registry.py` if you ship new Python helpers.
- New **patching primitive**: extend `src/autointerp/tools/head_patching.py`
  (per-head patching, ablation, path patching live there).

Smoke-test changes:

```bash
PYTHONPATH=src pytest tests/ -q
```
