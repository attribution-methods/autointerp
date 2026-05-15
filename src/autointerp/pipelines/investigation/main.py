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

from .report import write_report
from .run_dir import RunHandle, init_run, load_run
from .state import RunState, TerminalState, is_terminal_state, read_state


SYSTEM_PROMPT_HEADER = """\
You are the autointerp investigation agent. The user has approved an
InvestigationSpec; you are executing it.

# Inviolable rules

1. The spec is read-only. You may read it (`read_file spec.json`) but cannot
   edit it. Disagreements with the spec are resolved by `request_spec_revision`.
2. `compute_metric` is the only path to a `MetricResult`. Its value cannot be
   edited before commit; the gate verifies this via a one-time provenance
   token.
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
- Commit typed artifacts via `commit_artifact` (PromptBatch, ActivationCacheRef,
  CandidateSite, InterventionResult, …). The agent's `write_file` is range-
  restricted to `scripts/`, `scratch/`, and `INVESTIGATION_LOG.md`.
- When you finish a stage's metric work, run `evaluate_criterion` for any
  criterion that this stage's split now satisfies, then `advance_stage`.
- A criterion has three outcomes, not two. If the metric is computed but the
  evidence genuinely cannot support a PASS/FAIL (too few samples, wide CI,
  degenerate or contaminated data), pass `inconclusive_reason` to record it
  INCONCLUSIVE: the value is still logged, the run is NOT terminated, and the
  criterion counts as evaluated. Use this for honest non-results — not to
  dodge a FAIL you could defend. A FAIL still terminates the run.
- If results contradict the spec's hypothesis or methodology, do NOT silently
  reroute — call `request_spec_revision` with a clear `reason`.

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

Importable Python helpers under `autointerp.tools.*` (the repo root is on
sys.path; if not, `sys.path.insert(0, "src")` first). Use these from your
scripts before writing forward-pass / hook / patching code by hand:

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


def build_system_prompt(spec: InvestigationSpec, run_id: str) -> str:
    return (
        SYSTEM_PROMPT_HEADER
        + "\n\n"
        + f"# Run: {run_id}\n\n"
        + render_spec_summary(spec)
    )


def prepare_run(
    spec_path: str | Path,
    *,
    runs_root: Path | None = None,
    resume: bool | None = None,
) -> tuple[RunHandle, InvestigationSpec, RunState]:
    """Resolve the spec to either a fresh init_run or a resumed load_run.

    If ``resume`` is None: auto — resume if the run dir already exists.
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
    handle = init_run(spec, runs_root=runs_root)
    state = read_state(handle.state_path)
    return handle, spec, state


def is_terminal(state: RunState) -> bool:
    return is_terminal_state(state.terminal_state)


def finalize(handle: RunHandle) -> Path:
    """Write the final InvestigationReport and return its path."""
    return write_report(handle)


__all__ = [
    "SYSTEM_PROMPT_HEADER",
    "build_system_prompt",
    "finalize",
    "is_terminal",
    "prepare_run",
    "render_spec_summary",
]


# Touch-test imports that we want available re-exported from this module too.
_ = (TerminalState, textwrap, Any)
