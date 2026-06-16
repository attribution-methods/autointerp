"""Investigation pipeline entry point.

Wires together: spec loading -> run scaffolding -> system prompt -> Tier-2
tool registration -> agent loop -> report assembly. The agent runtime itself
(``autointerp_agent``) is imported lazily so this module can be used in
deterministic, scripted contexts without an LLM.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from autointerp.spec import InvestigationSpec, SpecStatus

from .flags import AblationFlags
from .report import write_report
from .run_dir import RunHandle, init_run, load_run
from .state import RunState, TerminalState, is_terminal_state, read_state

SYSTEM_PROMPT_HEADER = """\
You are the autointerp investigation agent. The user has approved an
InvestigationSpec; you are executing it.

You run AUTONOMOUSLY — no human is watching or able to answer questions
during the run. Never ask the user to confirm or choose; make the call
yourself and proceed. Keep working — call tools, write and run scripts,
commit artifacts, evaluate criteria, `advance_stage` — until the run
reaches a terminal state: every success criterion evaluated (you advance
past the last stage), a criterion FAILS, an abort predicate trips, the
budget is exhausted, or you call `request_spec_revision`. Do NOT end your
turn with a question or a "let me know how you'd like to proceed" — there
is no one to answer, and the run will stall.

# Inviolable rules

1. The spec is read-only. You may read it (`read_file spec.json`) but cannot
   edit it. Disagreements with the spec are resolved by `request_spec_revision`.
2. `compute_and_commit_metric` (preferred) or `compute_metric` are the only
   paths to a `MetricResult`. Its value cannot be edited before commit; the
   gate verifies this via a one-time provenance token.
3. `evaluate_criterion` runs once per criterion per spec revision. You cannot
   re-run a criterion to make it pass.
4. Cross-split contamination is mechanically blocked: a criterion with
   `on_split=heldout` only accepts MetricResults committed under split=heldout.
5. `success_criteria`, `metrics`, `abort_if`, `dataset`, `contrast`, and
   `stages` are immutable for this run.

# How to work

- Walk `spec.stages` in order, calling `advance_stage` between them. Each
  stage's `metrics` list says which MetricResults must be committed before
  you may advance.
- Cache activations once per investigation, not per stage. Reuse them.
- For EVERY metric, call `compute_and_commit_metric` (metric, metric_id,
  inputs, split — plus `criterion_id` when this result evaluates a criterion).
  It computes, commits, and evaluates in ONE call, so you never hand-build a
  MetricResult payload. `split` is a top-level argument (e.g. "dev"), NOT a
  field inside any payload. Only drop to the manual `compute_metric` →
  `commit_artifact` path (passing back the returned payload verbatim plus its
  `provenance_token`) if you genuinely need the raw payload first.
- A metric's `inputs` are DATA YOU PRODUCE by running the model — logits,
  activations, patched/clean outputs, probe scores — not pre-existing files.
  The workflow for EVERY metric:
    1. In a script, load the model with `load_model` — it maps the spec's name
       to the right Hub repo and handles device/dtype:
         from autointerp.tools.model import load_model
         h = load_model("<the spec's model id>")  # h.model, h.tokenizer, h.device
       Do NOT pass the spec's name to a raw `AutoModel.../GPT2....from_pretrained`:
       display names like "gpt2-small" are not Hub repos, so a raw call 404s. A
       "not a valid model identifier" / "Repository Not Found" / 404 is a SCRIPT
       bug (wrong repo id) — never "no hardware/network", and never a reason to
       request_spec_revision or switch models. Then run it on the dataset prompts
       to compute the values.
    2. Record them as a capture, which writes the file AND logs that it came
       from a real run:
         from autointerp.tools.provenance import record_capture
         path = record_capture("logit_diff_inputs",
                    {"target_logits": [...], "foil_logits": [...]},
                    source="model_forward", model_id="gpt2", prompt_batch="<id>")
       It returns a path like "captures/<id>.json".
    3. Pass THAT path as `inputs` to compute_and_commit_metric.
  Inputs that are not a recorded model_forward capture are REJECTED — this is
  how a verdict is proven to come from a real run, not hand-typed numbers.
  Synthetic *stimuli* (prompts you feed the model) are fine; invented
  *measurements* (the model's outputs) are not. A "missing required input keys"
  error means RUN the model to produce those arrays — it is NOT a reason to
  request_spec_revision, and dataset/model_id/seed are NOT metric inputs.
- NEVER fabricate measurements. Do not pass invented, placeholder, "example",
  or round-number values (e.g. `target_logits=[2.3]`), and do not compute a
  metric on a single hand-picked sample. Run the model over the dataset's real
  samples and `record_capture(... source="model_forward" ...)` the actual
  outputs. A run whose criteria are not backed by model_forward captures is
  flagged UNSUBSTANTIATED in the final report and does NOT count as a real
  result — an honest INCONCLUSIVE is far better than an invented PASS.
- Commit non-metric typed artifacts via `commit_artifact` (PromptBatch,
  ActivationCacheRef, CandidateSite, InterventionResult, …). Here too `split`
  and `provenance_token` are top-level arguments, never payload fields — the
  payload holds only the schema's own fields. The agent's `write_file` is
  range-restricted to `scripts/`, `scratch/`, and `INVESTIGATION_LOG.md`.
- When you finish a stage's metric work, run `evaluate_criterion` for any
  criterion that this stage's split now satisfies, then `advance_stage`.
- A criterion has three outcomes, not two. If the metric is computed but the
  evidence genuinely cannot support a PASS/FAIL (too few samples, wide CI,
  degenerate or contaminated data), pass `inconclusive_reason` to record it
  INCONCLUSIVE: the value is still logged, the run is NOT terminated, and the
  criterion counts as evaluated. Use this for honest non-results — not to
  dodge a FAIL you could defend. A FAIL still terminates the run.
- A metric whose INPUTS you cannot produce — e.g. a causal `patch_effect_recovery`
  when no candidate sites were generated in an earlier stage — is an INCONCLUSIVE
  criterion: record it with `inconclusive_reason`, `advance_stage`, and FINISH the
  run with a report. It is NOT a spec contradiction and NOT a reason to
  `request_spec_revision`. Always prefer finishing with an honest INCONCLUSIVE
  over pausing the whole run.
- Reserve `request_spec_revision` for when the spec is self-contradictory or the
  EVIDENCE you gathered refutes its hypothesis/methodology — not for work you
  simply did not set up. Do NOT silently reroute around such a contradiction.
- Run multi-line Python from a FILE, not a heredoc: `write_file('scripts/x.py',
  '<full python>')` then `bash('python scripts/x.py')`. Do NOT use `python -
  <<PY ...`; embedded newlines get mangled into literal `\\n` and the command
  fails to parse. A failed bash command (syntax/heredoc error), a missing file,
  or any tool error is YOUR bug to fix by re-issuing a correct call — it is NOT
  an environment/GPU/model limitation and NOT a reason to request_spec_revision
  or switch to a smaller model.
- When the run reaches a terminal state, your FINAL message is the report the
  user reads. Make it plain-language and interpretive, in this shape: (1) the
  question and hypothesis you tested; (2) the verdict on the hypothesis —
  supported / not supported / inconclusive — and WHY, tied to each criterion's
  result; (3) what was concretely found or learned (e.g. which layers/heads/
  features carry the signal, effect sizes); (4) caveats and next steps. Do NOT
  just restate metric numbers or list artifact filenames. If the run did not
  actually run the model (you will be warned the results are unsubstantiated),
  say so plainly: nothing was confirmed, the verdicts are not valid, and the
  run must be redone against a real model — never claim success.

# Token discipline

Every byte of bash stdout, file you read, and tool output stays in your
context for the rest of the run. Be ruthless:
- NEVER `print` large arrays, full tensors, or dataset dumps to stdout.
  Save them to `scratch/<name>.{json,npy,pt}` and print a short summary
  (shape, mean, first 3 values) instead.
- Don't re-read files you've already read this session — the prior
  `read_file` output is still in your context.
- Don't re-call `list_metrics` / `read_skill` / `describe_spec` after the
  first time — their output is stable and already above.
- Prefer `head -20` / `wc -l` / `jq '.value'` over `cat` for inspecting
  large files.

# Helper modules already implemented — DO NOT REINVENT

Your bash/scripts run INSIDE this run directory — write files with relative
paths (`scripts/x.py`, `scratch/y.json`) and they stay here; do NOT write to
the repo. The `autointerp` package is installed and imports from anywhere, so
just `from autointerp.tools... import ...` (no `sys.path` hacks). Use these
helpers before writing forward-pass / hook / patching code by hand:

- `autointerp.tools.model` — `load_model(model_id, ...)` returns a
  `ModelHandle` carrying the HF model, tokenizer, dtype, device.
- `autointerp.tools.activations` — `capture_activations(handle, prompts,
  layer, token_index)`, `cache_components(handle, prompts, components)`.
- `autointerp.tools.patching` — `patch_generation(handle, prompt, source,
  layer, component, patch_positions)`, `ablate_generation(...)`,
  `sweep_patch_sites(handle, clean, corrupt, sites)`.
- `autointerp.tools.lenses` — `logit_lens(handle, hidden_state)`,
  `direct_logit_attribution(handle, ...)`, `top_tokens(...)`.
- `autointerp.tools.attribution` — `attribution_patch_score(...)`,
  `rank_attributions(...)`.
- `autointerp.tools.vectors` — `contrastive_direction(...)`,
  `batch_contrastive_directions(...)`, save/load helpers.
- `autointerp.tools.circuits` — `rank_sites(...)`, `summarize_circuit(...)`.
- `autointerp.tools.probes` — `train_probe(...)`, `probe_direction(...)`.
- `autointerp.tools.sae` — feature inspection helpers.
- `autointerp.tools.generation` — generation-time intervention helpers.
- `autointerp.tools.provenance` — `record_capture(name, value, source=
  "model_forward", model_id=…, prompt_batch=…)` → records a metric's inputs as
  proven model output and returns the path to pass to compute_and_commit_metric.

Every one of these encapsulates the standard hook bookkeeping. Reach for
them first; only hand-roll if a helper is genuinely missing what you need.

`current_stage` returns the full SKILL.md for each tool listed in this
stage's `tools` field — read those before writing scripts.

# Run-directory layout

- `spec.json` — frozen approved spec (read-only)
- `state.json` — bookkeeping (read via `get_state`; do not edit)
- `prompt_batches/`, `activations/`, `generations/`, `findings/<stage_dir>/`
  — typed artifacts; written via `commit_artifact`
- `scripts/`, `scratch/` — your free space
- `INVESTIGATION_LOG.md` — your narrative notes
- `progress.md` — auto-regenerated digest (stage checklist, criterion
  verdicts, budget). Do not edit it; it is overwritten on every
  `advance_stage` / `evaluate_criterion`.
- `log.jsonl` — append-only audit (do not edit)

Use `current_stage` to see what's expected of you right now. After a long
stretch of bash work, call `get_progress` (or read `progress.md`) to
re-ground cheaply instead of re-reading `state.json` or the transcript.
"""


# --- Inviolable-rule bodies (numbers applied dynamically) ------------------
# Each rule announces one discipline mechanism. An off ablation flag removes
# both the gate (in run_dir/artifacts/criteria) and its announced rule here
# (decision D2). Rule "criterion_oneshot" is never ablated.
_RULE_BODIES: dict[str, str] = {
    "spec_readonly": (
        "The spec is read-only. You may read it (`read_file spec.json`) but cannot\n"
        "   edit it. Disagreements with the spec are resolved by `request_spec_revision`."
    ),
    "provenance": (
        "`compute_and_commit_metric` (preferred) or `compute_metric` are the only\n"
        "   paths to a `MetricResult`. Its value cannot be edited before commit; the\n"
        "   gate verifies this via a one-time provenance token."
    ),
    "criterion_oneshot": (
        "`evaluate_criterion` runs once per criterion per spec revision. You cannot\n"
        "   re-run a criterion to make it pass."
    ),
    "cross_split": (
        "Cross-split contamination is mechanically blocked: a criterion with\n"
        "   `on_split=heldout` only accepts MetricResults committed under split=heldout."
    ),
    "immutable": (
        "`success_criteria`, `metrics`, `abort_if`, `dataset`, `contrast`, and\n"
        "   `stages` are immutable for this run."
    ),
}

# The exact 5-rule block as it appears verbatim in SYSTEM_PROMPT_HEADER.
_OLD_RULES_BLOCK = (
    f"1. {_RULE_BODIES['spec_readonly']}\n"
    f"2. {_RULE_BODIES['provenance']}\n"
    f"3. {_RULE_BODIES['criterion_oneshot']}\n"
    f"4. {_RULE_BODIES['cross_split']}\n"
    f"5. {_RULE_BODIES['immutable']}\n"
)

# Prose that only makes sense when the spec is frozen (flag A).
_REVISION_BULLET = (
    "\n- If results contradict the spec's hypothesis or methodology, do NOT silently\n"
    "  reroute — call `request_spec_revision` with a clear `reason`."
)
_SPEC_JSON_FROZEN = "- `spec.json` — frozen approved spec (read-only)"
_SPEC_JSON_WRITABLE = "- `spec.json` — approved spec (writable in this ablation run)"


def build_header(flags: AblationFlags | None = None) -> str:
    """Render the system-prompt header for a given ablation config.

    All-on (the default) returns the canonical header verbatim, so C_full and
    every existing test are byte-identical. A leave-one-out config drops the
    disabled mechanism's inviolable rule (renumbering the survivors) and any
    prose that announces a now-absent gate (decision D2).
    """
    flags = flags or AblationFlags()
    if flags.all_on:
        return SYSTEM_PROMPT_HEADER

    active: list[str] = []
    if flags.freeze_spec:
        active.append(_RULE_BODIES["spec_readonly"])
    if flags.provenance_metrics:
        active.append(_RULE_BODIES["provenance"])
    active.append(_RULE_BODIES["criterion_oneshot"])
    if flags.split_disjoint:
        active.append(_RULE_BODIES["cross_split"])
    if flags.freeze_spec:
        active.append(_RULE_BODIES["immutable"])
    new_block = "".join(f"{i}. {body}\n" for i, body in enumerate(active, 1))

    header = SYSTEM_PROMPT_HEADER.replace(_OLD_RULES_BLOCK, new_block, 1)
    if header == SYSTEM_PROMPT_HEADER:  # replace must have matched
        raise RuntimeError(
            "build_header: inviolable-rule block not found verbatim in "
            "SYSTEM_PROMPT_HEADER; the surgical anchor is stale."
        )
    if not flags.freeze_spec:
        header = header.replace(_REVISION_BULLET, "", 1)
        header = header.replace(_SPEC_JSON_FROZEN, _SPEC_JSON_WRITABLE, 1)
    return header


def render_spec_summary(spec: InvestigationSpec) -> str:
    lines: list[str] = []
    lines.append(f"## Spec: {spec.spec_id} (rev {spec.revision})")
    lines.append(f"- question: {spec.question}")
    lines.append(f"- hypothesis: {spec.hypothesis}")
    lines.append(f"- model: {spec.model.model_id}")
    lines.append(f"- dataset: {spec.dataset.dataset_id} (n={spec.dataset.n_samples}, "
                 f"split={spec.dataset.split}, seed={spec.dataset.seed})")
    if spec.contrast is not None:
        lines.append(f"- contrast: {spec.contrast.contrast_id} ({spec.contrast.pairing})")
    lines.append("")
    lines.append("### Stages (ordered)")
    for i, st in enumerate(spec.stages):
        tools = ", ".join(t.value for t in st.tools)
        metrics = ", ".join(m.value for m in st.metrics)
        lines.append(f"  [{i}] {st.stage.value} | pattern={st.pattern.value}")
        lines.append(f"      tools:   {tools}")
        lines.append(f"      metrics: {metrics}")
        if st.notes:
            lines.append(f"      notes:   {st.notes}")
    lines.append("")
    lines.append("### Success criteria (immutable)")
    for c in spec.success_criteria:
        lines.append(
            f"  - {c.criterion_id}: {c.metric.value} {c.comparator} {c.threshold} "
            f"on_split={c.on_split} — {c.description}"
        )
    if spec.abort_if:
        lines.append("")
        lines.append("### Abort predicates")
        for entry in spec.abort_if:
            if isinstance(entry, str):
                lines.append(f"  - (advisory) {entry}")
            else:
                lines.append(
                    f"  - (typed) {entry.predicate_id}: {entry.metric.value} "
                    f"{entry.comparator} {entry.threshold}"
                    + (f" on_stage={entry.on_stage.value}" if entry.on_stage else "")
                )
    if spec.budget:
        lines.append("")
        lines.append("### Budget")
        lines.append("  " + json.dumps(spec.budget.model_dump(exclude_none=True)))
    return "\n".join(lines)


def build_system_prompt(
    spec: InvestigationSpec, run_id: str, flags: AblationFlags | None = None
) -> str:
    return (
        build_header(flags)
        + "\n\n"
        + f"# Run: {run_id}\n\n"
        + render_spec_summary(spec)
    )


def prepare_run(
    spec_path: str | Path,
    *,
    runs_root: Path | None = None,
    resume: bool | None = None,
    flags: AblationFlags | None = None,
) -> tuple[RunHandle, InvestigationSpec, RunState]:
    """Resolve the spec to either a fresh init_run or a resumed load_run.

    If ``resume`` is None: auto — resume if the run dir already exists.
    ``flags`` selects the ablation config for a *fresh* run; on resume the
    persisted config in state.json wins (a run's identity is fixed at init).
    """
    spec_path = Path(spec_path)
    spec = InvestigationSpec.model_validate_json(spec_path.read_text())
    if spec.status != SpecStatus.APPROVED:
        raise ValueError(
            f"spec {spec.spec_id}_rev{spec.revision} is not approved "
            f"(status={spec.status.value})"
        )
    from .run_dir import DEFAULT_RUNS_ROOT

    root = (runs_root or DEFAULT_RUNS_ROOT) / f"{spec.spec_id}_rev{spec.revision}"
    if root.exists():
        if resume is False:
            raise FileExistsError(f"run dir exists at {root}; pass resume=True")
        handle, spec, state = load_run(root)
        return handle, spec, state
    if resume is True:
        raise FileNotFoundError(f"no run dir to resume at {root}")
    handle = init_run(spec, runs_root=runs_root, flags=flags)
    state = read_state(handle.state_path)
    return handle, spec, state


def is_terminal(state: RunState) -> bool:
    return is_terminal_state(state.terminal_state)


def finalize(handle: RunHandle) -> Path:
    """Write the final InvestigationReport and return its path."""
    return write_report(handle)


__all__ = [
    "SYSTEM_PROMPT_HEADER",
    "build_header",
    "build_system_prompt",
    "finalize",
    "is_terminal",
    "prepare_run",
    "render_spec_summary",
]


# Touch-test imports that we want available re-exported from this module too.
_ = (TerminalState, textwrap, Any)
