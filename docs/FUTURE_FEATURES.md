# Future Features

A thorough, opinionated punch-list of things to build on top of the current
autointerp scaffold. Organized by leverage. Items flagged *(known gap)* are
already called out in `docs/investigation.md` or `docs/IOI_END_TO_END_REPORT.md`
as parked / deferred; everything else is fresh.

## Top 5 — highest leverage this week

1. **Fix the "agent reinvents the metric-commit dance" friction.** ✅ **DONE** —
   `compute_and_commit_metric(metric, metric_id, inputs, split, criterion_id?)`
   in `pipelines/investigation/metrics.py` composes `compute_metric` →
   `commit_artifact` → (optional) `evaluate_criterion` and returns
   `{metric_result, artifact_ref, criterion_record}`. Exposed as a Tier-2
   tool and listed before `compute_metric` in the registered set so the
   agent prefers it. Lower-level primitives stay available. 6 new tests in
   `tests/test_compute_and_commit.py`.
2. **Wire range-restricted `write_file`.** ✅ **DONE** — `ToolRouter.writable_roots`
   is opt-in (None for Stage 0 / REPL, set to `RunHandle.writable_roots()` by
   the investigation CLI). `_write_file` and `_edit_file` route through
   `autointerp.utils.paths.normalize_safe_path` + `is_within`, so paths
   outside `scripts/`, `scratch/`, and `INVESTIGATION_LOG.md` are refused
   with a clear error pointing at `commit_artifact`. 12 new tests in
   `tests/test_writable_roots.py`.
3. **Fill in canonical impls for the other registered metrics.** ✅ **DONE** —
   `auroc` (Mann-Whitney U with midrank tie handling), `hit_rate` (mean of
   `hits`), `necessity_drop` (`clean - with_component_removed`),
   `completeness` (`circuit_only / full`), and `sufficiency`
   (`only_component / full`) all have canonical impls in
   `pipelines/investigation/metrics.py`. The registry now covers 13 of the
   23 closed-vocabulary metrics; the rest (`direct_contribution`,
   `paraphrase_invariance`, `mutual_information`, `causal_indirect_effect`,
   `feature_*`, `steering_effect_size`) need domain-specific input
   primitives or model internals and are deferred. 20 new tests in
   `tests/test_metric_impls.py`.
4. **Auto-revision loop.** ✅ **DONE** — `cmd_investigate` now loops up to
   `--max-revisions N` (default 3): when a run terminates as
   `REVISION_REQUESTED` or `CRITERION_FAILED`, it builds a Markdown
   summary of the prior run via `revision_loop.summarize_for_stage0`
   (terminal state, criteria evaluated, abort/revision reason, log tail,
   findings list) and re-engages Stage 0 with that summary preloaded so
   the planner can draft a child spec citing `parent_spec_id` and
   `prior_results_ref`. Other terminal states (`COMPLETED`, `ABORTED`,
   `BUDGET_EXHAUSTED`) end the work. 7 new tests in
   `tests/test_revision_loop.py`.
5. **Behavior-bench with ≥4 published targets.** *(known gap, partial.)*
   IOI is solved-by-construction; the literature has dozens of canonical
   mechanisms. Pick: induction-head detection (small models), factual
   recall S-inhibition (Pythia), refusal direction (Llama-3-Instruct),
   greater-than circuit (GPT-2). Score: did the agent recover the published
   heads/layers/directions? This becomes the regression test.

## Pipeline & runtime robustness

6. **Bash watchdog / liveness.** ✅ **DONE** — `_run_bash_watched` in
   `autointerp_agent/tools.py` polls stdout via non-blocking `os.read`,
   kills the process on a no-output stall (`stall_seconds`, default 600s)
   *and* on hard `timeout`, surfaces non-zero exits with `[exit N]` prefix,
   and streams full output to `<scratch_dir>/bash_<pid>.log` for live
   tailing. Investigation CLI passes `handle.scratch_dir`. 6 new tests in
   `tests/test_bash_watchdog.py`.
7. **Real budget instrumentation.** `gpu_seconds`, `wallclock_seconds`,
   `samples` are recorded as `0.0` in the attention_concentration run —
   nothing populates them. Either delete those budget keys or instrument
   them (`time.monotonic()` around bash; CUDA event timers around HF model
   calls). Without this, `BUDGET_EXHAUSTED` can never trip on compute, only
   tool-call count.
8. **Cost telemetry.** ✅ **DONE** — `autointerp.utils.cost.CostTracker`
   accumulates per-model tokens (input/output, cache read/write) and USD
   from LiteLLM responses (`_hidden_params.response_cost`, falling back to
   `litellm.completion_cost`). `run_agent_turn` accepts an optional
   `cost_tracker`; the investigation CLI persists `cost.json` in the run
   dir (atomic write, restored on resume keyed by `run_id`) and threads
   the snapshot into `report.metadata["cost"]`. 6 new tests in
   `tests/test_cost_tracker.py`.
9. **Content-addressed activation cache shared across runs.** *(known
   gap.)* Currently each `runs/<spec>` re-extracts the same activations.
   Hash `(model_id, dataset_hash, layer_spec)` → store under
   `~/.cache/autointerp/activations/<hash>/`; symlink into per-run dirs.
   Big speedup when the same agent reruns a similar spec.
10. **Determinism check.** Add `runs verify <run_id>` that re-executes the
    committed metric scripts against the on-disk inputs and asserts values
    match within a seed-window. Catches silent regressions when shipping
    new tools.
11. **Run-dir curation.** `runs/` already has 9 entries including
    `_failed_*`, `_haiku_*`, `_stale_smoke_*`. Add `runs gc` (move to
    `runs/_archive/` after N days unless `terminal_state == COMPLETED`).
12. **Per-criterion `evaluate_after_stage_idx`.** *(known gap — deferred.)*
    `on_split` inference works but is implicit. One field, one validator
    line, eliminates a class of "criterion evaluated too early" footguns.

## Method-coverage gaps

13. **Edge Attribution Patching (EAP).** Faster than activation patching for
    triage, much more accurate than gradient×activation, and natural fit
    for the existing `attribution-patching` skill. Especially useful at
    >7B scale where head sweeps blow up.
14. **Multi-step path patching.** `head_patching.path_patch` does
    single-step; the IOI literature uses 3+ step path patching to
    decompose chains. Generalize to a path object:
    `Path(senders=..., receivers=..., freeze_others=True)`.
15. **Pretrained-SAE loaders.** GemmaScope, Llama-Scope, JumpReLU SAEs are
    on the hub. `load_pretrained_sae(model, layer, release="gemma-scope-9b-pt-res")`
    eliminates the "you'd have to train an SAE first" objection for 90% of
    investigations.
16. **SAE feature → causal effect ranking.** Current SAE tools are
    lookup-only. Combine with attribution patching on SAE features (the
    standard cross-layer transcoder recipe) to rank features by causal
    effect on a behavior metric. This is *the* feature-discovery pipeline.
17. **Standard attention-pattern detectors.** Attention-pattern tools were
    added in `12903a6`; package the recipes — `is_induction_head`,
    `is_copy_head`, `is_s_inhibition_head`, `attention_to_position(p)`.
    Each is one cell in a notebook today; promote to skill+tool.
18. **Cross-attention / per-position interventions.** *(known gap — called
    out as thin in IOI report.)* Current patching API takes a layer/head;
    extend to `(layer, head, src_pos, dst_pos)` for per-edge attention
    surgery.
19. **Refusal-direction recipe.** Mean-diff between harmful/harmless on
    instruct models, ablate the direction, eval on held-out — *the*
    most-replicated 7B-scale investigation in the literature. Should be a
    one-shot pipeline pattern: `pattern: refusal_direction_ablation`.
20. **Activation-oracle / verbalizer wiring.** The `activation-oracles`
    skill exists but no implementation. Wire an explainer-model loop
    (Sonnet/Haiku looks at top-activating examples, proposes a label) —
    closes the SAE-feature labeling story.
21. **Counterfactual / contrast-pair generator.** Stage 0 currently asks
    the user to provide the contrast template. A tool that *proposes*
    clean/corrupt templates from the behavior description (LLM-generated,
    schema-validated) would cut Stage 0 length in half.

## Stage 0 quality (the long-pole conversation)

22. **Spec linter (advisory, not blocking).** "Clean and corrupt prompts
    have a length mismatch — confounds attention-pattern claims."
    "Hypothesis is unfalsifiable — propose a falsification condition."
    "abort_if is empty — under what evidence would you give up?" Not
    mechanical errors, but cheap to detect and high-value.
23. **Sample-size / power calculator tool.** Given comparator + threshold +
    expected effect size, propose `dataset.size`. The agent currently
    picks `500 / 100` heuristically.
24. **Pattern auto-suggestion.** From the behavior text, infer a likely
    `PatternId` (`blackbox_then_patching`, `refusal_direction`,
    `induction_detection`, …). Today the agent picks; meta-knowing which
    pattern fits the question would inform stage selection.
25. **Phenomenon registry / `priors/` populated.** That dir exists but is
    empty. Each known phenomenon (IOI, induction, refusal, factual recall)
    gets a markdown card with: typical model, typical layers, typical
    metrics, gotchas. The agent reads via `read_phenomenon` during Stage
    0 — saves rediscovering the wheel.
26. **Hypothesis schema typing.** Current `hypothesis: str`. Promote to a
    tagged union (`HypothesisRoute`, `HypothesisAdditive`,
    `HypothesisInhibition`, …) so the planner can flag "your stages don't
    actually test this hypothesis type."
27. **Stage 0 max-turns: surface where it stalled.** `--max-turns` exists;
    if it trips, dump the partial draft + last 3 turns to a structured
    "what's missing" file the user can complete by hand.

## Evaluation: how do you know it's improving

28. **No-gating ablation.** *(known open question.)* Run the same spec
    with all Tier-2 gates replaced by free `commit_artifact` (no
    provenance token, no canonical metrics). Compare: how often do values
    disagree with re-derived ground truth? Quantifies what gating actually
    buys.
29. **Negative-control specs.** Specs for behaviors that *don't* exist in
    the chosen model (e.g. "GPT-2-small implements arithmetic
    chain-of-thought"). Correct outcome: `request_spec_revision` with a
    sound rationale. If criteria pass, the gates are too lax.
30. **Adversarial / red-team specs.** A "p-hacker" prompt: tell the agent
    to find *any* significant pattern, no pre-registration. Compare to
    the gated-spec version on the same model. Quantifies the
    gating-vs-no-gating delta.
31. **Reproducibility eval.** Run the same approved spec twice with
    different seeds. How often do criteria flip? If it's >5%, thresholds
    are too tight relative to seed variance.

## Multi-agent / parked ablations

32. **Per-stage agent handoff.** *(parked future ablation.)* Agent context
    length will bite at >7B-scale runs. Per-stage agents handing off via
    `INVESTIGATION_LOG.md` plus a structured "stage exit memo" gives
    bounded context per agent.
33. **Stage-end auto-summarization.** Even with one agent, compress at
    stage boundaries: produce `findings/stage_<idx>/SUMMARY.md` that the
    next stage's prompt starts with. Today the agent rebuilds context
    implicitly; explicit handoff is cheaper.
34. **"Suggested pipeline" mode.** *(parked future ablation.)* Allow
    agent-driven stage reordering with `revision_reason`; useful when an
    early result obviates a later stage. Today the agent must walk all of
    `spec.stages` even if stage 3's metric already settles the question.

## Observability & UX

35. **`autointerp runs replay <run_id>`** — re-execute the agent's
    tool-call trace deterministically (no LLM calls). Surfaces
    non-determinism in tools; lets you debug a run for free.
36. **Live web dashboard.** Even a static-HTML render of the run dir
    (criteria table, transcript stream, stage timeline) would be more
    usable than `runs tail`.
37. **Notebook export.** Convert a finished run to a Jupyter notebook with
    each `findings/*` as a cell + matching `scripts/*.py` as runnable
    reproducer cells. Pairs well with publishing a run as a writeup.
38. **`autointerp runs diff <run_a> <run_b>`** — show which
    criteria/findings differ between two runs of the same spec. Essential
    when comparing models or seeds.

## Safety / methodological rigor

39. **Independent criterion auditor.** A tool that, given a finished run,
    re-evaluates every success criterion *from scratch* against the
    committed artifacts, ignoring `state.criteria_evaluated`. If the agent
    or gate had a bug, this catches it. ~50 lines.
40. **Provenance for activation tensors.** `cache_id` is in the JSON
    manifest, but the `.pt` files don't carry the hash. A bit-flip in
    storage is silently accepted. Add a manifest hash, verify on read.
41. **Disjoint-split contamination tests.** Split tags are enforced at
    commit time; add a periodic check that no committed `MetricResult`
    accidentally references inputs from a different split's prompt batch.
42. **Pre-registration audit log.** `log.jsonl` is nice but doesn't make
    the *predictions* explicit. A `report.predictions` section that, at
    run start, snapshots `(criterion_id → predicted pass/fail)` based on
    the hypothesis lets you score the agent's calibration.

## Scale-up

43. **Llama-3.1-8B and Qwen-2.5-7B** as second + third targets. *(known
    gap.)* Most published interp work is on these now; testing only on
    GPT-2-small is a streetlight effect.
44. **API-only end-to-end demo.** `black-box-auditing` skill exists, but
    no run is purely API-only. Pick a real target (sycophancy on Claude
    API, or jailbreak-susceptibility on a specific model), run it, and
    add the report to `docs/`.
45. **Multi-model A/B.** Same spec, two models. Does Llama implement IOI
    the same way as GPT-2? Publishable, and tests the toolkit's
    portability.

## Long shots / creative

46. **Auto-skill-bundle inference.** Today the user has to know which
    skills bundle to load. The Stage-0 planner could pick from a library
    of "skill bundles" tagged by phenomenon family.
47. **Spec-from-paper.** Drop a PDF; produce a Stage-0 spec that aims to
    replicate the paper's headline mechanism. (Realistic for 1–2 page
    interp papers.)
48. **Counterfactual circuit narration.** Once a circuit is found, the
    agent generates a natural-language story ("L9H9 attends from END to
    IO and copies its embedding into the output projection") and then
    *predicts* a new prompt where the circuit will fail. If the prediction
    lands, strong evidence; if not, a self-detected gap.
49. **Cross-run learning.** A `findings/` index across all runs that the
    planner reads at Stage 0: "the last 4 specs that targeted
    attention_concentration on Llama-3 found L17H7; consider it as a
    prior."
50. **MCP packaging.** Expose the Tier-2 gates as an MCP server so the
    same investigation pipeline can be driven by Claude Desktop, Codex,
    or any other MCP-aware host. The schema's portability is wasted as
    long as the gates are CLI-only.

---

## Suggested first slice

Three items to ship cleanly:

1. **`compute_and_commit_metric` wrapper + range-restricted `write_file`** —
   biggest agent-quality gain per line of code.
2. **Auto-revision loop** — closes the loop the REPL already half-implements.
3. **Behavior-bench with one new published target** (induction or refusal) —
   gives ground truth to measure the other two against.
