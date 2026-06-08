# Stage 0 — Conversational Investigation Specification

Stage 0 turns a vague research question ("how does the model do X?") into a
**pre-registered, falsifiable, machine-readable** investigation plan that
downstream stages can consume. The agent and the user co-design an
`InvestigationSpec` across turns; on approval, the spec is frozen and written
to `outputs/specs/<spec_id>_rev<n>.json`.

## Why pre-registration

Mechanistic interpretability claims are easy to p-hack: discovery and
validation often share data, metrics get redefined after seeing results, and
"the circuit" changes shape to fit each new ablation. Stage 0 forces the
question, hypothesis, dataset, metrics, and success thresholds to be written
down **before** any expensive run. After approval, success criteria can only
change through a new spec revision — they are immutable on the current rev.

## The artifact

The contract is `autointerp.spec.InvestigationSpec`. Top-level fields:

- `spec_id`, `revision`, `parent_spec_id`, `prior_results_ref`,
  `revision_reason` — identity and the spec-revision DAG.
- `question`, `hypothesis`, `phenomenon_id` — what's being asked and the
  prediction.
- `behavior` (`BehaviorSpec`), `model` (`ModelRef`), `dataset` (`DatasetSpec`),
  `contrast` (`ContrastSpec` | None) — the experimental setup.
- `stages: list[StageSpec]` — ordered methodological pipeline.
- `success_criteria: list[Criterion]` — pre-registered, falsifiable thresholds.
- `abort_if`, `max_revisions`, `budget`, `risks` — safety and scope.
- `status` (`SpecStatus`), `approval` (`Approval` | None) — workflow state.

Stages, criteria, tools, metrics, and patterns all reference closed
vocabularies (enums) declared in `autointerp.spec`. Inventing values is
rejected at write time, not at finalize time.

## Closed vocabularies

| Enum | What it constrains | Where defined |
| --- | --- | --- |
| `MetricName` | Allowed metrics in `Criterion.metric` and `StageSpec.metrics` | `spec.py` |
| `MetricFamily` | Behavioral / localization / causal / generalization / feature / information | `spec.py` |
| `ToolName` | Methodological tools a stage may use | `spec.py` |
| `PatternId` | Named investigation patterns (`blackbox_then_patching`, etc.) | `spec.py` |
| `InvestigationStage` | Stage type (setup, black_box, localization, intervention, validation, …) | `spec.py` |

Each enum value carries machine-readable metadata:

- `METRIC_META[name]` → `MetricMeta(family, value_range, direction, requires_inputs, one_line)`
- `TOOL_META[name]` → `ToolMeta(requires_fields, families_emitted, one_line)`

The validator and the pydantic model validators read this metadata to enforce
contracts (range, direction, tool→required-field deps).

## Layers of validation

Three layers, each at a different time:

1. **Prevention at write time** — `update_spec` rejects unknown top-level
   keys, with suggestions ("did you mean `phenomenon_id`?"). This stops the
   agent from polluting the draft with hallucinated fields.

2. **Per-field schema (pydantic)** — types, enums, required fields,
   `Criterion.threshold ∈ metric.value_range`,
   `Criterion.comparator` direction matches `metric.direction`,
   `metric=CUSTOM` requires `custom_metric_def`,
   spec-level invariants (revisions need a parent, approved specs carry an
   `Approval`, every tool's `requires_fields` is set on the spec).

3. **Cross-field validator (`validate_spec` tool)** — wraps `try_build` and
   pretty-prints pydantic errors so the agent can self-check mid-conversation.
   No advisory warnings — methodological judgment (sample size, threshold
   tuning, dataset choice) belongs to the user, not to code.

The agent fixes mechanical errors silently. The user reviews methodology when
the rendered spec is presented for approval.

## The agent's tool surface

| Tool | Purpose |
| --- | --- |
| `describe_spec` | One-shot schema reference: fields, types, required/optional, enum values |
| `update_spec` | Patch top-level fields. Rejects unknown keys at patch time |
| `remove_spec_fields` | Delete top-level keys from the draft |
| `show_spec` | Render the full target shape; unset fields show as `_(unset)_` |
| `validate_spec` | Run pydantic validation on the partial draft |
| `list_metrics` | Browse `MetricName` (optionally filtered by family) |
| `read_metric` | Reference card + contract for a single metric |
| `finalize_spec` | Two-phase approval; writes the approved spec to disk |

`describe_spec` is generated from the pydantic models, so it can never drift
from the schema. `read_metric` reads from `metrics/<name>.md` plus the typed
`MetricMeta` contract.

## Persistence

Every `update_spec` snapshots the partial draft to
`outputs/specs/_draft.json`. Any Stage 0 tool call on a fresh process reloads
it on first use, so an interrupted conversation can be resumed across
sessions. `finalize_spec` writes the approved spec and clears the draft.

## Approval flow

`finalize_spec` is two-phase by design:

1. **Phase 1** (`user_confirmed` unset) — returns the rendered spec.
   The agent shows it to the user verbatim and asks for approval.
2. **Phase 2** (`user_confirmed=true`) — runs `try_build` one last time,
   constructs the `Approval` record, freezes status to `approved`, writes
   `outputs/specs/<spec_id>_rev<n>.json`, and clears the draft.

Phase 2 fails loudly on any structural error. The user-facing surface stays
clean: spec → "approve?" → "approved." All mechanical concerns happen
between the agent and the tools.

## Spec revisions

After Stage 1+ produces results, a child spec can be drafted with
`parent_spec_id`, `prior_results_ref`, and `revision_reason` set. The schema
enforces that a revision points at its parent. `SpecRevisionLink` records
the edge with the outcome that motivated the child. (Tooling for the
revise-on-results loop is not yet implemented; the schema fields are.)

## Conversational design

The Stage 0 system prompt (in `configs/agent.yaml`) is intentionally short.
It gives the agent character ("Work like a careful research engineer") and
soft rules ("Build the spec INCREMENTALLY", "explain your reasoning to the
user as you go") rather than a turn-by-turn template. Mechanical lookups
(`describe_spec`, `list_metrics` / `read_metric`) and silent error-handling
are the only hard rules.

The user is the **methodological reviewer**. Code catches what code can prove
wrong; the user judges sample size, dataset choice, threshold tuning, and
hypothesis quality when the rendered spec is presented.

## Quick start

```bash
PYTHONPATH=src python -m autointerp_agent \
  --model anthropic/claude-sonnet-4-6 --auto-approve
```

Then ask a research question, e.g. *"how does the model perform IOI?"*. The
agent will recommend a model, walk through behavior framing → contrast →
dataset → hypothesis → stages → metrics, present the rendered spec, and
finalize on your approval. The approved spec lands in `outputs/specs/`.
