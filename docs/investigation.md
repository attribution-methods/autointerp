# Investigation Pipeline — Design Notes

This document records the design decisions for the *investigation pipeline*:
the system that takes an approved `InvestigationSpec` (produced by Stage 0,
see [stage0.md](stage0.md)) and executes it.

Naming: Stage 0 builds an `InvestigationSpec`; the **investigation pipeline**
executes it and produces an `InvestigationReport`. Code lives at
`src/autointerp/pipelines/investigation/`.

Status: **steps 1–8 implemented (115 tests passing). Step 9 (real model run on
`do-identification-pythia-410m-v1_rev1`) still TODO.**

## Locked-in decisions

1. **Coding agent, not deterministic dispatcher.** The agent has free-form
   `bash` / `read_file` / `write_file` / `edit_file` and writes the code that
   produces evidence. Pre-registration is enforced by a tool gate, not by
   restricting agent behavior.

2. **One long-lived agent per spec rev.** The agent walks `spec.stages` in
   order via `advance_stage()`. Single LiteLLM process, full spec visibility,
   shared `INVESTIGATION_LOG.md` for narrative continuity.

3. **`spec.stages` is binding.** The agent must execute each entry in order.
   `tools` and `metrics` listed in a `StageSpec` constrain what it can use
   and what artifacts it must produce in that stage. Deviation requires a
   spec revision.

4. **Pre-registration enforcement lives in the tool gate.** The agent sees
   the full spec; it just cannot bypass:
   - canonical metric computation (`compute_metric`);
   - one-shot criterion evaluation (`evaluate_criterion`);
   - split-tagged artifacts (heldout-only criteria refuse non-heldout inputs);
   - read-only frozen `spec.json`.

5. **State persistence is a single mutable `state.json` (atomic writes).**
   No `state_history.jsonl`. `log.jsonl` already provides per-tool-call audit.

6. **Range-restricted `write_file`.** In the investigation pipeline, the
   agent's `write_file` tool can only write under `scripts/`, `scratch/`, and
   `INVESTIGATION_LOG.md`. Typed artifacts go through `commit_artifact`.

7. **Provenance tokens for `MetricResult`.** `compute_metric` issues a
   one-time token; `commit_artifact("MetricResult", ...)` requires it. Closes
   the "agent makes up a number" hole.

## Parked as future ablations

Keep the architecture flexible enough that these can be swapped in without
rewrites:

- **One-agent-per-stage** with handoff via `INVESTIGATION_LOG.md`. Implication:
  current-stage selection must be a single piece of `state.json`, not baked
  into the agent process.
- **"Suggested pipeline" mode** where the agent may deviate from
  `spec.stages` with a `revision_reason`. Implication: stage advancement is
  via a tool, not implicit.
- **Auto-injecting SKILL.md** for the current stage's `tools` into the system
  prompt. For now: lazy via `read_skill`, with a one-line hint pointing the
  agent at relevant skills.

## Architecture

```
   Stage 0 (existing)
        │  approved InvestigationSpec
        ▼
   investigation.run(spec_path)
        │
        ▼
   init_run / load_run  ──▶  runs/<spec_id>_rev<n>/
        │
        ▼
   single agent loop (LiteLLM)
     system prompt = constraints + spec + schema ref + run-dir guide
     tools         = Tier 1 (free-form) + Tier 2 (gated)
        │
        ├── advance_stage() iterates spec.stages
        ├── abort_if / budget guards may terminate
        └── request_spec_revision() is the deviation exit
        │
        ▼
   report.json (InvestigationReport, assembled from artifacts)
```

## Code layout

```
src/autointerp/pipelines/investigation/
  __init__.py
  run_dir.py        # init/load run, paths, chmod-readonly spec.json
  state.py          # state.json schema (pydantic) + atomic read/write
  artifacts.py      # commit_artifact gate (typed, range-restricted)
  metrics.py        # compute_metric registry (canonical impls)
  criteria.py       # evaluate_criterion (one-shot per spec rev)
  guards.py         # abort_if predicates + budget middleware
  stages.py         # advance_stage, current_stage_view
  tools.py          # Tier 2 tool wrappers (LLM-facing)
  main.py           # CLI: investigation.run(spec_path)
```

The agent runtime stays in `src/autointerp_agent/` and imports from here.
A new entry point analogous to Stage 0:

```bash
python -m autointerp_agent.investigation \
  --spec outputs/specs/<spec_id>_rev<n>.json
```

## Run directory layout

```
runs/<spec_id>_rev<n>/
  spec.json                          # frozen copy of approved spec (chmod r--)
  state.json                         # mutable bookkeeping (atomic writes only)
  log.jsonl                          # append-only audit of every tool call
  INVESTIGATION_LOG.md               # agent's narrative notes (free-form)

  prompt_batches/
    <batch_id>.json                  # PromptBatch (typed)

  activations/
    <cache_id>.json                  # ActivationCacheRef (no tensors)
    <cache_id>/                      # tensors on disk
      manifest.json                  # layer/component/dtype/shape index
      <layer>_<component>.pt         # tensors

  generations/
    <sample_id>.json                 # GenerationSample

  findings/
    stage_<idx>_<name>/              # one dir per executed spec stage
      behavioral_<id>.json           # BehavioralFinding
      candidate_<id>.json            # CandidateSite
      feature_<id>.json              # FeatureFinding
      intervention_<id>.json         # InterventionResult
      validation_<id>.json           # ValidationResult
      metric_<id>.json               # MetricResult

  scripts/                           # whatever Python the agent writes
  scratch/                           # free-play; not consumed by report

  assistant_turns.jsonl              # every assistant LLM message (full content + tool-call summaries)
  tool_invocations.jsonl             # every tool call: full args + output preview + body_ref
  tool_invocations/                  # full untruncated output body per call
    <iter>_<idx>_<tool>_<call_id>.txt

  report.json                        # final InvestigationReport
```

Three rules that fall out of this layout (each one is load-bearing):

1. `spec.json` is `chmod r--` after `init_run`. Read-only via `read_file`.
2. Everything in `prompt_batches/`, `activations/`, `generations/`,
   `findings/` is append-only and only writable via `commit_artifact`.
3. `state.json` and `log.jsonl` are written only by the gate. The agent has
   no tool that touches them directly.

## state.json schema (sketch)

```jsonc
{
  "schema_version": 1,
  "spec_id": "...",
  "spec_revision": 1,
  "run_id": "<spec_id>_rev<n>",
  "run_started_at": "ISO-8601",
  "run_ended_at": null,

  "current_stage_idx": 1,
  "stage_status": {
    "<idx>": {
      "stage": "localization",
      "status": "pending|in_progress|completed|aborted",
      "started_at": "...",
      "ended_at": null,
      "abort_reason": null,
      "artifact_refs": ["findings/stage_0_black_box/metric_accuracy-001.json", ...]
    }
  },

  "criteria_evaluated": {            // write-once per (criterion_id, spec_rev)
    "<criterion_id>": {
      "passed": true,
      "value": 0.83,
      "metric": "accuracy",
      "comparator": ">=",
      "threshold": 0.8,
      "metric_result_ref": "findings/.../metric_*.json",
      "evaluated_at": "..."
    }
  },

  "abort_triggered": null,           // or {predicate_id, metric, value, at_stage_idx, at}

  "budget_consumed": {
    "tool_calls": 0, "bash_calls": 0,
    "gpu_seconds": 0.0, "wallclock_seconds": 0.0, "samples": 0
  },

  "spec_revision_requested": null,   // or {reason, prior_results_ref, requested_at}

  "terminal_state": null,            // null | "completed" | "criterion_failed"
                                      //      | "aborted" | "budget_exhausted"
                                      //      | "revision_requested"

  "pending_provenance_tokens": {     // issued by compute_metric, consumed by commit_artifact
    "<token>": {
      "metric": "accuracy",
      "metric_id": "acc-001",
      "value": 0.83,
      "inputs_hash": "sha256:...",
      "issued_at": "...",
      "stage_idx": 0
    }
  },
  "provenance_tokens_consumed": 0
}
```

Gate invariants over this state:

- `current_stage_idx` advances monotonically; `advance_stage` refuses unless
  criteria scoped to the current stage are evaluated and passing.
- `criteria_evaluated[criterion_id]` is write-once. Re-call returns cached.
- `pending_provenance_tokens` is one-time-use; consumption deletes the entry.
- `terminal_state != null` ⇒ all non-read-only tool calls return an error.

## log.jsonl entry shape

```json
{
  "ts": "ISO-8601",
  "stage_idx": 1,
  "tool": "compute_metric",
  "args": {"name": "patch_effect_recovery", "inputs_hash": "sha256:..."},
  "ok": true,
  "result_summary": "value=0.42; provenance_token=tok_a1f3...",
  "budget_after": {"tool_calls": 15, "gpu_seconds": 415.1}
}
```

Bash entries log the command, exit code, and stdout/stderr lengths only.
Output bodies, if needed, go to `scratch/` via `write_file`.

## Tool surface

**Tier 1 — free-form (reuse existing where possible):**
`bash`, `read_file`, `write_file` (range-restricted), `edit_file`,
`list_skills`, `read_skill`, `glob`, `grep`.

**Tier 2 — gated (new):**
- `commit_artifact(kind, payload, provenance_token?)` — pydantic-validates
  against `schemas.py`, tags `(stage_idx, split)`, writes to the run dir.
  `MetricResult` requires `provenance_token`.
- `compute_metric(name, inputs)` — canonical impl per `MetricName`. Returns
  `(payload, provenance_token)`. Records seed and inputs_hash.
- `evaluate_criterion(criterion_id)` — one-shot per spec rev. Refuses
  cross-split contamination.
- `advance_stage()` — closes current stage; runs `abort_if`; evaluates
  stage-scoped criteria; refuses on missing artifacts or failures.
- `current_stage()` — read-only view: current `StageSpec`, allowed tools,
  expected metrics, notes, list of artifacts committed this stage.
- `get_state()`, `get_budget()` — read-only.
- `request_spec_revision(reason, prior_results_ref)` — terminal; punts back
  to Stage 0.

## Provenance token mechanics

1. `compute_metric(name, inputs)` runs the canonical implementation, hashes
   `(metric_name, normalized_inputs_json, run_id)`, registers a token in
   `state.pending_provenance_tokens`, returns `(payload, token)` to the agent.
2. `commit_artifact("MetricResult", payload, provenance_token=tok)` validates
   that `tok` exists and that `payload.metric_id` / `payload.value` match
   what the registry recorded, then consumes the token.
3. Agents cannot produce a token any other way; state.json is unreachable to
   them.

~30 lines of code, closes the entire "fabricate a number" path.

## Resume semantics

`init_run` on an existing dir → load `state.json`, present the agent with:
- rendered spec,
- summary of completed stages + artifact refs,
- current stage view (what's left),
- `INVESTIGATION_LOG.md` for prior reasoning.

Mid-bash crashes are idempotent: bash output is never an artifact until
`commit_artifact` runs. `state.json` writes are atomic (write-temp-then-
rename), so partial states aren't observable.

## Schema extensions to Stage 0 (small pre-reqs)

- **Type `abort_if`.** Currently `list[str]` (LLM-evaluated). Promote to
  `list[AbortPredicate]` analogous to `Criterion` (`metric`, `comparator`,
  `threshold`, `on_stage`). Keep accepting the old string form as a
  deprecation transition. Without typing, the abort gate is meaningless.
- *(deferred)* Per-`Criterion` field naming the spec stage index it should
  be evaluated after. Currently inferred from `on_split`; works but implicit.
- *(deferred)* Top-level `seed` registry beyond `dataset.seed`. For now,
  derive intervention/ablation seeds by hashing `(dataset.seed, stage_idx,
  op_name)`.

## Build order

Each step CI-runnable on its own; the GPU only enters at step 9.

1. `run_dir.py` + `state.py` — init/load/atomic-write/resume. Unit-testable.
2. `artifacts.py` — `commit_artifact` gate. Round-trip every type from
   `schemas.py`.
3. Type `abort_if` in `spec.py`; migrate. Update Stage 0 `describe_spec`.
4. `metrics.py` — canonical implementations for `accuracy`, `logit_diff`,
   `faithfulness` (the three the Pythia approved spec uses). Provenance
   tokens.
5. `criteria.py` — one-shot evaluator. Tests: re-call returns cached;
   wrong-split refused.
6. `guards.py` — typed `abort_if` evaluator + budget middleware.
7. `tools.py` — Tier 2 wrappers exposed to LLM.
8. `main.py` + system prompt + agent entry point. End-to-end smoke test on a
   *fixture* model (mirror `examples/blackbox_to_validation.py`).
9. Real run on `do-identification-pythia-410m-v1_rev1`.

## What's actually built (vs. the design)

- **`run_dir.py` + `state.py`** — `RunHandle`, `init_run`, `load_run`,
  atomic `state.json` read/write. ``spec.json`` chmod r-- after init.
- **`artifacts.py`** — `commit_artifact` gate over all 9 schema types.
  Append-only, split-tagged via ``metadata['_provenance']``, refuses overwrite,
  refuses post-terminal commits, integrates budget + abort guards.
- **Stage 0 schema extension** — `AbortPredicate` (typed, range-validated,
  optional `on_stage`) added to `spec.py`; `abort_if: list[AbortPredicate | str]`
  is backwards-compatible with the existing approved Pythia spec.
- **`metrics.py`** — canonical implementations for `accuracy`, `logit_diff`,
  `faithfulness` (the three the Pythia approved spec uses). Range-clipping
  records the unclipped value in metadata. Provenance tokens with
  `metric_id` + `value` + `inputs_hash` + `stage_idx` close the
  fabricate-a-number hole end-to-end.
- **`criteria.py`** — one-shot `evaluate_criterion` with split-disjointness
  enforcement (auto-discovery + optional explicit `metric_result_ref`); a
  failed criterion flips the run to `CRITERION_FAILED`.
- **`guards.py`** — `check_abort_predicates` (run after every `MetricResult`
  commit), `enforce_budget` (run at every gated entry).
- **`stages.py`** — `advance_stage` (refuses on missing metrics or tripped
  guards; final advance requires all criteria evaluated; rolls back on
  unevaluated), `current_stage_view`.
- **`revision.py`** — `request_spec_revision` (only legitimate exit when
  spec is wrong; allowed from `CRITERION_FAILED`, blocked from other
  terminal states).
- **`report.py`** — `assemble_report` / `write_report` produce a
  schema-valid `InvestigationReport` from on-disk artifacts.
- **`tools.py`** — Tier-2 LLM-facing `ToolSpec`s: `commit_artifact`,
  `compute_metric`, `evaluate_criterion`, `advance_stage`, `current_stage`,
  `get_state`, `get_budget`, `request_spec_revision`.
- **`main.py`** — `prepare_run` (init or resume with `resume=True/False/auto`),
  `build_system_prompt`, `render_spec_summary`, `finalize`.
- **`autointerp_agent/investigation.py`** — CLI:
  ``python -m autointerp_agent.investigation --spec <path>``.

## Deferred

These are intentionally NOT in the current build:

- **Range-restricted `write_file`** — design doc still calls for it (only
  `scripts/`, `scratch/`, `INVESTIGATION_LOG.md` writable). The pipeline
  exposes `RunHandle.writable_roots()` for the future restrictor; the agent
  runtime's `write_file` is currently *not* wired through it. Lowest-risk
  follow-up.
- **Watchdog for long-running bash jobs** — see MEMORY note about prior
  pain with nohup pipelines lying about completion. Not yet wired in.
- **Auto-injected SKILL.md per stage** — still lazy via `read_skill`.
- **Step 9 real run** — pythia-410m on the approved spec. Needs a GPU and
  the `mechinterp` extra installed.

## Implementation reference

This section documents what the gates *actually do*. Function signatures
mirror what's in code; error types are what callers (the agent loop, scripted
tests, future ablations) catch.

### `pipelines.investigation.run_dir`

```python
init_run(spec: InvestigationSpec, runs_root: Path | None = None) -> RunHandle
load_run(run_dir: Path) -> tuple[RunHandle, InvestigationSpec, RunState]
```

- `init_run` refuses non-approved specs (`status != APPROVED`) and refuses
  if the run directory already exists (use `load_run` to resume).
- `init_run` writes a frozen copy of `spec.json` and `chmod 0o444` it.
- `load_run` re-asserts the read-only mode on resume (idempotent), creates
  any missing scaffolding, and validates that `state.json` identity matches
  the on-disk spec.
- `RunHandle` carries every path the gates need: `spec_path`, `state_path`,
  `log_path`, `narrative_path`, `prompt_batches_dir`, `activations_dir`,
  `generations_dir`, `findings_dir`, `scripts_dir`, `scratch_dir`,
  `report_path`. `writable_roots()` lists the future agent-only writable
  paths (`scripts/`, `scratch/`, `INVESTIGATION_LOG.md`).

### `pipelines.investigation.state`

```python
read_state(path: Path) -> RunState
write_state(path: Path, state: RunState) -> None      # atomic: tmp → fsync → rename
initial_state(spec_id, spec_revision, n_stages, stage_names) -> RunState
```

`RunState` is pydantic with `extra="forbid"`. Subrecords:

- `StageRecord(stage, status, started_at, ended_at, abort_reason, artifact_refs)`
- `CriterionRecord(passed, value, metric, comparator, threshold, metric_result_ref, evaluated_at)`
- `AbortRecord(predicate_id, metric, value, at_stage_idx, at)`
- `BudgetConsumed(tool_calls, bash_calls, gpu_seconds, wallclock_seconds, samples)`
- `RevisionRequest(reason, prior_results_ref, requested_at)`
- `ProvenanceToken(metric, metric_id, value, inputs_hash, issued_at, stage_idx)`
- `TerminalState ∈ {COMPLETED, CRITERION_FAILED, ABORTED, BUDGET_EXHAUSTED, REVISION_REQUESTED}`

### `pipelines.investigation.artifacts`

```python
commit_artifact(
    handle: RunHandle,
    kind: str,
    payload: dict | BaseModel,
    *,
    split: str | None = None,
    provenance_token: str | None = None,
) -> ArtifactRef
```

Raises `ArtifactGateError`. Order of checks:

1. `enforce_budget(handle)` — refuses if budget exhausted (re-raised from
   `GuardError`).
2. `state.terminal_state` must be `None`.
3. Validate payload via the schema class registered in `ARTIFACT_KINDS`.
4. Resolve split: `PromptBatch` reads `payload.split`; everything else
   requires the explicit `split=` arg.
5. Resolve target path; refuse if file already exists (append-only).
6. For `MetricResult`: require `provenance_token`, verify token's
   `metric_id`, `value`, and `stage_idx` match the payload and current
   stage; reject `provenance_token` for any other `kind`.
7. Stamp `metadata["_provenance"] = {stage_idx, split, committed_at}`.
   Reject payloads that pre-set `_provenance` (reserved namespace).
8. Atomic file write; flip `StageStatus.PENDING → IN_PROGRESS`; consume
   token if applicable; increment `budget_consumed.tool_calls`; append a
   `log.jsonl` entry; for `MetricResult`, run `check_abort_predicates`.

`ARTIFACT_KINDS` covers: `PromptBatch`, `ActivationCacheRef`,
`GenerationSample`, `BehavioralFinding`, `CandidateSite`, `FeatureFinding`,
`InterventionResult`, `ValidationResult`, `MetricResult`.

### `pipelines.investigation.metrics`

```python
compute_metric(
    handle: RunHandle,
    *,
    metric: MetricName | str,
    metric_id: str,
    inputs: dict[str, Any],
    threshold: float | None = None,
    comparator: str | None = None,
    metadata: dict | None = None,
) -> tuple[dict, str]   # (MetricResult payload dict, provenance_token)
```

Raises `MetricRegistryError`. Currently registered:

| Metric | Required inputs | Returned value |
| --- | --- | --- |
| `accuracy` | `predictions`, `labels` (lists, equal length, non-empty) | `mean(p == l)` ∈ [0, 1] |
| `logit_diff` | `target_logits`, `foil_logits` (lists of floats, equal length) | `mean(t - f)` ∈ ℝ |
| `faithfulness` | `circuit_metric`, `full_model_metric`, `corrupted_metric` | `(circuit − corrupted) / (full − corrupted)`, clipped to [0, 1] (unclipped value preserved in metadata) |

Other `MetricName` values raise "no canonical implementation" — extend the
registry by PR.

The returned payload is a `MetricResult` dict with:
- `metric_id`, `value` (post-clip), `threshold`, `passed` (if threshold +
  comparator given), `metadata.metric_name`, `metadata.metric_family`,
  `metadata.inputs_hash`, `metadata.registry_version`, optionally
  `metadata.unclipped_value` and `metadata.clipped_to_range`.

Provenance tokens are `tok_<32hex>`. They are single-use; consumption by
`commit_artifact` increments `state.provenance_tokens_consumed` and removes
the entry from `pending_provenance_tokens`.

### `pipelines.investigation.criteria`

```python
evaluate_criterion(
    handle: RunHandle,
    criterion_id: str,
    *,
    metric_result_ref: str | None = None,   # path under run root, or None for auto-find
) -> CriterionRecord
```

Raises `CriterionGateError`. Behavior:

- Re-call returns the cached `CriterionRecord`.
- Without `metric_result_ref`: auto-finds committed `MetricResult` artifacts
  whose `metadata.metric_name` and `metadata._provenance.split` match the
  criterion. Refuses with a `multiple match` error if more than one
  candidate exists.
- With `metric_result_ref`: refuses if the referenced artifact's metric or
  split tag does not match the criterion (split-disjointness enforcement).
- `passed=False` flips `terminal_state → CRITERION_FAILED` and sets
  `run_ended_at`. `request_spec_revision` is the only path forward.

### `pipelines.investigation.guards`

```python
check_abort_predicates(handle: RunHandle) -> AbortRecord | None
enforce_budget(handle: RunHandle) -> None     # raises GuardError on exhaustion
```

- `check_abort_predicates` walks `typed_abort_predicates(spec)`, scans
  committed `MetricResult` artifacts, and trips the run with
  `terminal_state = ABORTED` on the first matching `(predicate, artifact)`
  pair where the comparator fires. Idempotent. `on_stage` filters
  predicates to a specific spec stage by index lookup.
- `enforce_budget` checks `spec.budget` against
  `state.budget_consumed`; on exhaustion, flips to `BUDGET_EXHAUSTED` and
  raises `GuardError`. Wrapped at the entry of `compute_metric` (re-raised
  as `MetricRegistryError`) and `commit_artifact` (re-raised as
  `ArtifactGateError`).

### `pipelines.investigation.stages`

```python
advance_stage(handle: RunHandle) -> dict[str, Any]
current_stage_view(handle: RunHandle) -> dict[str, Any]
```

Raises `StageGateError`. `advance_stage` does:

1. `enforce_budget` + terminal-state check.
2. For the current stage, every `MetricName` in `stage.metrics` must have
   at least one committed `MetricResult` whose `metadata.metric_name`
   matches.
3. `check_abort_predicates` — refuses if anything has tripped.
4. Mark stage `COMPLETED`, set `ended_at`, increment `current_stage_idx`.
5. If now past the last stage: every `success_criterion.criterion_id` must
   appear in `state.criteria_evaluated`. Missing ⇒ rollback (stage stays
   `IN_PROGRESS`, `current_stage_idx` reverts) and refuse with
   "unevaluated".
6. If terminal: set `terminal_state = COMPLETED` (unless something earlier
   set it to `CRITERION_FAILED`).

`current_stage_view` returns `{stage_idx, n_stages, stage, pattern, tools,
metrics, notes, status, artifact_refs, committed_metric_names, complete,
terminal_state}`.

### `pipelines.investigation.revision`

```python
request_spec_revision(
    handle: RunHandle,
    *,
    reason: str,
    prior_results_ref: str | None = None,
) -> RevisionRequest
```

Raises `RevisionGateError`. Allowed from `terminal_state ∈ {None,
CRITERION_FAILED}`; blocked from `COMPLETED`, `ABORTED`, `BUDGET_EXHAUSTED`,
or already-requested. Idempotent: a second call returns the first record.
Sets `terminal_state = REVISION_REQUESTED`.

### `pipelines.investigation.report`

```python
assemble_report(handle: RunHandle) -> InvestigationReport
write_report(handle: RunHandle) -> Path                      # writes report.json
```

`assemble_report` walks every typed artifact subdir, validates each via the
matching `schemas.py` class, and constructs a single `InvestigationReport`
pinning the spec id/revision, terminal state, all evaluated criteria, and
`budget_consumed` into `report.metadata`. Pure read on the run dir.

### `pipelines.investigation.main`

```python
prepare_run(
    spec_path: str | Path,
    *,
    runs_root: Path | None = None,
    resume: bool | None = None,    # None = auto, True = require existing, False = require fresh
) -> tuple[RunHandle, InvestigationSpec, RunState]

build_system_prompt(spec: InvestigationSpec, run_id: str) -> str
render_spec_summary(spec: InvestigationSpec) -> str
finalize(handle: RunHandle) -> Path
```

The system prompt embeds the rendered spec verbatim — gate enforcement
makes opacity unnecessary, and full visibility lets the agent plan caching
across stages.

### `pipelines.investigation.observer`

```python
class RunObserver:
    def __init__(self, handle: RunHandle, *, console: Any | None = None) -> None: ...
```

Persistent transcript writer + optional live console stream. Attaches to
``run_agent_turn(..., observer=...)`` and writes:

- ``assistant_turns.jsonl`` — one line per assistant LLM message (full
  content + tool-call summaries). Final assistant message gets
  ``"final": true``.
- ``tool_invocations.jsonl`` — one line per tool call: ``args``,
  ``ok``, ``output_chars``, ``output_preview`` (first ~400 chars), and
  ``body_ref`` pointing at the full body file.
- ``tool_invocations/<iter>_<idx>_<tool>_<call_id>.txt`` — full
  untruncated output body for every call (so bash stdout/stderr,
  long error traces, raw JSON tool returns are all replayable).

Complementary to ``log.jsonl`` (which only records *gated tool calls*
that hit ``compute_metric`` / ``commit_artifact`` / etc). The observer
captures everything the agent actually did.

Observer hooks never propagate exceptions back into the agent loop — a
broken console or full disk degrades transcript quality but cannot
crash the run.

### `pipelines.investigation.tools`

```python
create_investigation_tools(handle: RunHandle) -> list[ToolSpec]
```

Returns 8 `ToolSpec`s using the existing `autointerp_agent.tools.ToolSpec`
shape so they register cleanly into the existing `ToolRouter`. Each handler
catches its module-level gate exception and returns `(message, ok=False)`
so the agent loop sees structured errors rather than raised exceptions.

### Stage 0 schema additions (`autointerp.spec`)

```python
class AbortPredicate(StrictBaseModel):
    predicate_id: str
    description: str
    metric: MetricName
    comparator: Literal[">=", ">", "<=", "<", "=="]
    threshold: float
    on_stage: InvestigationStage | None = None
    metric_params: dict[str, Any] = Field(default_factory=dict)
```

`InvestigationSpec.abort_if: list[AbortPredicate | str]` accepts the typed
form (mechanically evaluated) and the legacy free-text form (advisory only;
the existing approved Pythia spec, with two string entries, still loads
unchanged).

Helper: `typed_abort_predicates(spec) -> list[AbortPredicate]` filters out
the advisory entries — the gate iterates over this.

## Agent CLI

```bash
python -m autointerp_agent.investigation \
  --spec outputs/specs/<spec_id>_rev<n>.json \
  [--runs-root runs] \
  [--prompt "..."] \
  [--model anthropic/claude-...] \
  [--auto-approve] \
  [--resume | --no-resume] \
  [--quiet]                         # disable live console streaming (transcripts still written to disk)
```

Flow:

1. Resolves the spec → `prepare_run` (auto-resume by default).
2. Builds the system prompt via `build_system_prompt`.
3. Spawns `ToolRouter` with the existing Tier-1 tools (`bash`, `read_file`,
   `write_file`, `edit_file`, `list_skills`, `read_skill`, …) plus the
   Tier-2 set from `create_investigation_tools(handle)`.
4. Attaches a `RunObserver` (live console + persistent transcript) so every
   assistant turn and tool call lands on disk for post-hoc debugging.
5. Runs one agent turn. If `state.terminal_state` is set on exit, writes
   `report.json`.

A run can be resumed by re-invoking with the same `--spec` (and matching
`--runs-root`); the state survives full process restarts via the atomic
`state.json` writes.

## Open items / followups

- Range-restricted `write_file` — `RunHandle.writable_roots()` is ready;
  the agent runtime's `write_file` is not yet wired through it.
- Watchdog for long-running bash jobs (see MEMORY note about prior pain
  with nohup pipelines lying about completion).
- Per-`Criterion` field naming the spec stage index it should be evaluated
  after (currently inferred from `on_split`; works but implicit).
- Top-level `seed` registry beyond `dataset.seed` — for now derived seeds
  via `hash(dataset.seed, stage_idx, op_name)`.
- Canonical implementations for the rest of `MetricName` (currently:
  `accuracy`, `logit_diff`, `faithfulness`).
- Reference cards in `metrics/<name>.md` for every registered metric.
