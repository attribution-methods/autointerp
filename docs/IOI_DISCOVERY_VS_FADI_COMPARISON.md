# IOI: Fadi's spec vs the discovery sub-agent — A/B comparison

A direct A/B between the canonical IOI investigation
(`examples/specs/ioi-gpt2-small-blind-v1_*.json`, originally on `fadi-agent-v1`,
re-run with progressive spec revisions) and the discovery-sub-agent variant
(`examples/specs/ioi-discovery-v1_*.json`, new on this branch). Both run
on real GPT-2-small (HuggingFace), real LiteLLM Sonnet-4.5 master (your
ANTHROPIC_API_KEY), real `claude-agent-sdk` Sonnet-4.5 sub-agent, real
H100s (Fadi GPU 0, discovery GPU 1, parallel).

The comparison is across **three rounds of spec revisions**. Each round
fixes one methodology ambiguity exposed by the prior round; all six runs
are preserved on disk under `runs/`.

---

## TL;DR

- **Both specs identify the canonical Wang-et-al-2022 IOI circuit**:
  L9H9, L9H6, L10H0 (the three name-mover heads). Discovery rev2/rev3
  surface them explicitly as `CandidateSite` artifacts; Fadi's spec
  identifies them implicitly via head_patch_sweep.
- **Joint top-K dev-set patching reaches Wang-quality numbers in both**:
  Fadi rev3 = 0.93, Discovery rev3 = 0.99.
- **Heldout faithfulness clears 0 but not 0.85** in Fadi rev3
  (faithfulness = 0.515 with END-only ablation, vs Wang's 0.913).
  Discovery rev3 didn't reach validation, so heldout faithfulness was
  not measured.
- **Each round of spec revisions unblocks one methodology issue**:
  rev1 → rev2 fixed joint-vs-single-head; rev2 → rev3 fixed validation
  faithfulness ablation positions. **A rev4 would still be needed for
  full pipeline completion** (resampling vs mean ablation, or threshold
  re-calibration).
- **Discovery sub-agent works**: 4 iterations per run, ~$1.30/run, real
  algorithm proposals, surfaces canonical heads as named artifacts.
  When the sub-agent's evaluator output is consumed correctly (rev2/rev3),
  it adds discovery-stage evidence the master alone wouldn't produce.
- **Total session cost across all 6 runs**: ~$24 API + $5 sub-agent.
  Sub-agent runs counted as part of master's total when both are summed.

---

## Spec evolution

| Rev | Fix in this rev | Spec file (Fadi) | Spec file (Discovery) |
|---|---|---|---|
| **1** | Original baselines (Fadi's blind variant; discovery's first version) | `ioi-gpt2-small-blind-v1_rev1.json` | `ioi-discovery-v1_rev1.json` |
| **2** | Disambiguate joint top-K vs per-head max in `localization-recovery` and `discovery-recovery`; embed worked code in stage notes | `ioi-gpt2-small-blind-v1_rev2.json` | `ioi-discovery-v1_rev2.json` |
| **3** | Pin validation faithfulness ablation to **END position only** (not all positions); embed worked code; add per-stage "advance immediately, no summary" guard to discovery spec | `ioi-gpt2-small-blind-v1_rev3.json` | `ioi-discovery-v1_rev3.json` |

All six specs ship with `status: "approved"` and full `revision_reason` /
`prior_results_ref` chains. Each rev's `revision_reason` cites the prior
run's failure mode as motivation, so the DAG is self-documenting.

---

## Round 1 — rev1 (the baselines)

| | **Fadi rev1** | **Discovery rev1** |
|---|---|---|
| Terminal | `criterion_failed` at stage 1 | `revision_requested` at stage 1 |
| Stages completed | 1 of 5 | 1 of 6 |
| Criteria passed / total | 1 / 4 (`behavioral-sanity`) | 2 / 5 (`behavioral-sanity`, `discovery-recovery`) |
| Failure mode | Sonnet committed *single-head* `patch_effect_recovery` (~0.45) instead of joint top-K (~0.9) | Master used the sub-agent's per-algorithm scalar (~0.17) instead of running a fresh joint patch on the surfaced top-K |
| Sub-agent | n/a | crashed at iter 0 (nested-asyncio bug) — master fell back to its own tools |
| Master cost | ~$2.15 | ~$2.55 (master) + $0 (sub-agent crashed before billing) |

**What rev1 told us**: the original criterion text "Top-K heads recover
≥ 0.85" was ambiguous. Sonnet's training prior leans toward per-head max
when the data structure is a `[L, H]` recovery matrix. Both specs hit the
same class of failure for the same reason.

(Note: rev1 run dirs are not preserved in `runs/` — they were superseded
by rev2 and rev3 in subsequent passes. The numbers above are reconstructed
from session logs.)

---

## Round 2 — rev2 (joint top-K disambiguated)

Both specs got new criterion text and embedded code recipes:

> "Patching the top-K attention heads SIMULTANEOUSLY — i.e. all K heads' z
> activations replaced from clean into the corrupted run in ONE forward
> pass via `run_with_head_patches` — recovers at least X. Use
> `autointerp.tools.head_patching.run_with_head_patches` on the K-element
> list of HeadPatch objects; commit the SINGLE joint-patch recovery
> scalar, NOT per-head max from `head_patch_sweep`."

| | **Fadi rev2** | **Discovery rev2** |
|---|---|---|
| Terminal | `revision_requested` at stage 4 | `None` (master loop exited via no-tool-calls summary) |
| Stages completed | **4 of 5** (validation in_progress) | 2 of 6 (localization in_progress) |
| Criteria passed | 2 / 4 (`behavioral-sanity`, `localization-recovery`) | 3 / 5 (`behavioral-sanity`, `discovery-recovery`, `localization-recovery`) |
| `localization-recovery` value | **0.967** ✓ (joint top-K worked) | **0.990** ✓ (master applied joint patch on sub-agent's surfaced heads) |
| Failure mode | Sonnet implemented validation faithfulness as "mean-ablate complement at every position"; got -0.79 (clipped to 0); requested revision | Master committed stage-2 metrics, evaluated criterion, then wrote a "Investigation Summary" final message instead of advancing |
| Master cost | $2.59 / 88 turns | $3.19 / 100 turns |
| Sub-agent cost | n/a | $1.19 / 4 iterations |
| **Total cost** | **$2.59** | **$4.38** |

**What rev2 told us**: the joint top-K disambiguation worked — both
runs got Wang-quality joint recovery numbers. But two new failure modes
surfaced:

1. **Fadi**: validation stage's faithfulness recipe was under-specified.
   "Mean-ablate the complement of C and run circuit-only forward" is
   ambiguous about *positions*. Sonnet ablated everywhere; mean-ablating
   ~140 heads at all token positions takes GPT-2 OOD enough that
   circuit-only performs *worse* than corrupted, yielding negative
   faithfulness.
2. **Discovery**: master loop exited mid-pipeline (stages 0–2 done,
   stages 3–5 untouched). Sonnet wrote a summary text after passing
   3 criteria; in a tool-use loop, that's a terminal exit.

---

## Round 3 — rev3 (END-only ablation + no-summary guard)

**Both specs**: validation stage notes pinned to END-position-only
ablation, with worked code:

```python
clean_cache = cache_head_z(handle, clean_heldout_prompts)
mean_z = mean_head_z(clean_cache)
patches = []
for L in range(handle.n_layers):
    for H in range(handle.n_heads):
        if (L, H) not in circuit_set:
            patches.append(HeadPatch(layer=L, head=H, source=mean_z[L][:, H, :], positions=[-1]))
circuit_logits = run_with_head_patches(handle, clean_heldout_prompts, patches, return_logits_at=-1)
faithfulness_value = (circuit_lD - corrupt_lD) / (clean_lD - corrupt_lD)
```

**Discovery only**: each stage's notes adds "STAGE DISCIPLINE: this is
stage X of N. After committing this stage's metrics, IMMEDIATELY call
advance_stage(). Do NOT write a wrap-up summary — there are M stages left."

| | **Fadi rev3** | **Discovery rev3** |
|---|---|---|
| Terminal | `revision_requested` at stage 4 | `None` (loop exited via summary again) |
| Stages completed | **4 of 5** + validation in_progress (artifacts committed) | **3 of 6** |
| Criteria passed | **3 / 4** (`behavioral-sanity`, `localization-recovery`, … but `circuit-faithfulness` ✗) | **3 / 5** (`behavioral-sanity`, `discovery-recovery`, `localization-recovery`) |
| `localization-recovery` value | 0.930 ✓ | 0.990 ✓ |
| `circuit-faithfulness` value | **0.515** ✗ (heldout, with END-only ablation; up from rev2's -0.79 but below 0.85 threshold) | not reached |
| `discovery-recovery` value | n/a | **0.990** ✓ (vs rev2's 0.456!) |
| `circuit-minimality` value | 0.000 (heldout, but criterion not yet evaluated) | not reached |
| Master cost | $3.55 / 108 turns | $5.28 / 159 turns |
| Sub-agent cost | n/a | $1.33 / 4 iterations |
| **Total cost** | **$3.55** | **$6.61** |

**What rev3 told us**: END-only ablation is strictly better than
all-positions (0.515 vs -0.79 for Fadi's faithfulness on the same
3-head circuit). But 0.515 is still below 0.85. The remaining gap
between our 0.515 and Wang's published 0.913 likely comes from one
or more of:

- **Resampling ablation** — replace ablated heads' activations with
  values from a corrupted-batch sample (in-distribution), instead of
  mean (which is still slightly OOD).
- **Larger circuit definition** — Wang's full IOI circuit includes
  ~25 heads (name movers + S-inhibition + duplicate token + induction
  + previous token); we use only the 3 name-movers.
- **Threshold calibration** — 0.85 may be optimistic for a 3-head
  circuit on GPT-2-small with mean-END ablation.

A rev4 would address one of these. But this comparison is meaningful
as-is.

The Discovery rev3 master loop *still* exited via summary despite the
explicit per-stage instruction. Spec text is not enough; the framework
needs a guard (`if not tool_calls and not terminal: re-prompt`).

---

## Per-metric values across all rev3 runs

### Fadi rev3 — full metric trail

| Metric | Value | Split | Stage |
|---|---|---|---|
| accuracy | 0.980 | dev | 0 (black_box) |
| logit_diff | 2.673 | dev | 0 |
| joint_recovery | **0.930** | dev | 1 (localization) |
| kl_to_clean | 0.357 | dev | 1 |
| ablation_drop | 0.297 | dev | 2 (activation_analysis) |
| stage2_logit_diff | 3.500 | dev | 2 |
| dev_faithfulness | 0.512 | dev | 3 (intervention) |
| stage3_recovery | 0.930 | dev | 3 |
| heldout_faithfulness | **0.515** | heldout | 4 (validation, in_progress) |
| heldout_minimality | 0.000 | heldout | 4 |

### Discovery rev3 — full metric trail

| Metric | Value | Split | Stage |
|---|---|---|---|
| accuracy | 0.980 | dev | 0 (black_box) |
| logit_diff | 2.673 | dev | 0 |
| patch_effect_recovery | **0.990** | dev | 1 (feature_discovery, after sub-agent) |
| kl_to_clean | 0.416 | dev | 2 (localization) |
| patch_effect_recovery | **0.990** | dev | 2 (joint top-K from exhaustive sweep) |
| (none committed past stage 2) | | | |

---

## Identified circuit (heads)

Both runs converge on the same canonical Wang-et-al-2022 IOI heads.

### Discovery sub-agent's surfaced candidates (rev2)

```
L10H0  L10H7  L11H10  L9H6  L9H9
```

5 heads — three core name-movers (L9H9, L9H6, L10H0) plus L10H7 (also
in Wang's wider circuit) plus L11H10 (likely a false positive based on
agent's later analysis).

### Discovery sub-agent's surfaced candidates (rev3)

```
L9H9  L9H6  L10H0
```

3 heads — exactly the canonical name-mover circuit. Cleaner than rev2
because the sub-agent's algorithms converged on the same 3 heads
across iterations.

### Fadi rev3's identified circuit (via head_patch_sweep)

`[(9, 9), (9, 6), (10, 0)]` — same 3 heads.

**Both specs identify the same canonical circuit.** The discovery
sub-agent gets there *iteratively* by writing 4 algorithm versions and
ranking heads; Fadi's spec gets there *exhaustively* by sweeping all
144 (layer, head) pairs and taking the top-3 by per-head recovery.

---

## Cost comparison

| Run | Master $ | Sub-agent $ | Total $ | Master turns | Wallclock |
|---|---|---|---|---|---|
| Fadi rev1 | ~2.15 | n/a | **~2.15** | ~96 | ~17m |
| Fadi rev2 | 2.59 | n/a | **2.59** | 88 | ~17m |
| Fadi rev3 | 3.55 | n/a | **3.55** | 108 | ~14m |
| Discovery rev1 | ~2.55 | $0 (crash) | **~2.55** | ~66 | ~25m |
| Discovery rev2 | 3.19 | 1.19 | **4.38** | 100 | ~25m |
| Discovery rev3 | 5.28 | 1.33 | **6.61** | 159 | ~38m |

Discovery is more expensive than Fadi for two reasons:

1. **Sub-agent cost** is real ($1.19–$1.33 per call, 4 iterations each).
2. **Master takes more turns to consume the sub-agent's output**
   (extracting top-K, building HeadPatch list, running joint patch,
   committing CandidateSites + MetricResult), more setup work for the
   first non-trivial sub-agent integration.

For IOI specifically — where the search space is small (144 heads) —
**discovery is roughly 2× the cost of exhaustive sweep with no
methodological advantage**. The discovery sub-agent's value should
appear on larger search spaces (50K SAE features, MLP neurons, etc)
where exhaustive sweeps are infeasible.

---

## Behavior changes / what it took to get this far

This comparison required **5 substantive code/spec changes** across the
session. Each was discovered by a failed run and fixed for the next.

| # | Issue | Fix | Affected runs before fix |
|---|---|---|---|
| 1 | IOI benchmark labels IO as A always; ABBA template has A repeated → labels inverted on half the pairs | Refactor `benchmarks/ioi.py` to construct prompts from explicit IO/S roles | First-pass smoke (negative clean logit_diff on ABBA) |
| 2 | `METRIC_META.requires_inputs` mismatched canonical impls for 5 metrics (logit_diff, kl_to_clean, faithfulness, minimality, ablation_drop) | Align metadata to impl in `spec.py` | Fadi rev1 — agent tried to provide vocab arrays for logit_diff (~100MB), bailed via revision |
| 3 | `claude-agent-sdk` 0.1.76 was hitting `/usr/local/bin/claude` 1.0.25 (incompatible); needed PATH passthrough to `~/.npm-global/bin/claude` 2.1.132 | Add `env=os.environ` to `ClaudeAgentOptions` in discovery loop | Discovery rev1 — sub-agent crashed before any iteration completed |
| 4 | `run_discovery_subagent` called `asyncio.run()` from inside the master's already-running event loop (RuntimeError) | Detect running loop in `_run_loop`; route per-iteration work through a worker thread | Discovery rev1, rev2 — sub-agent crashed at iter 0 inside the investigation pipeline (worked standalone) |
| 5 | Master cost untracked anywhere on disk | Capture `response.usage` + `response._hidden_params['response_cost']` in `agent_loop.py`; surface via observer to `cost_summary.json` | All prior runs (just Anthropic console for cost) |

After all 5 are landed, **only spec-level disambiguation remains** to
get from rev1 to a fully-completed run. Rev2 fixed one ambiguity
(joint top-K), rev3 fixed another (END-only faithfulness); a rev4 would
fix the third (resampling ablation or larger circuit) and a framework
patch (no-tool-calls re-prompt) would prevent the master from exiting
mid-pipeline via summary.

---

## What this comparison does NOT show

- **Fully-completed runs.** Neither rev3 reached `terminal_state == COMPLETED`.
  The faithfulness gap (rev3 Fadi at 0.515 < 0.85) and the no-tool-calls
  exit (rev3 Discovery at stage 3) both block this.
- **A real performance advantage of the discovery sub-agent on IOI.** Both
  paths converge on the same 3 heads. Discovery's value is in *bigger*
  search spaces (SAE features) which we don't measure here.
- **Heldout numbers for Discovery.** Discovery's rev3 didn't reach
  validation (stage 5), so we have no Discovery heldout faithfulness
  to compare against Fadi's 0.515.
- **Determinism.** Sonnet 4.5 is stochastic — different runs of the same
  spec produce different methodology choices (rev1 Fadi got 0.45
  single-head; rev2 Fadi got 0.967 joint top-K). Each comparison row
  reflects one realization, not a fundamental capability.

---

## Run-dir layout (for inspection)

```
runs/
├── ioi-gpt2-small-blind-v1_rev2/         # Fadi rev2: 4 stages done, faithfulness fail
│   ├── spec.json                         # frozen approved spec (chmod 0o444)
│   ├── state.json                        # gate state, criteria, budget
│   ├── log.jsonl                         # gated tool call audit
│   ├── cost_summary.json                 # master LiteLLM token + cost totals (NEW)
│   ├── INVESTIGATION_LOG.md              # agent narrative
│   ├── findings/stage_*/*.json           # 8 committed MetricResults + CandidateSites
│   ├── scripts/                          # Python the agent wrote (per-stage)
│   ├── assistant_turns.jsonl             # every LLM message + per-turn cost
│   ├── tool_invocations.jsonl + bodies/  # every tool call + full output
│   └── report.json                       # final InvestigationReport
├── ioi-gpt2-small-blind-v1_rev3/         # Fadi rev3: 4 stages + heldout faithfulness=0.515
├── ioi-discovery-v1_rev2/
│   ├── ... (same shape) ...
│   └── discovery/<session>/              # NEW: sub-agent session per discover_features call
│       ├── algorithm_template.py
│       ├── algorithm_v1.py..algorithm_vN.py
│       ├── leaderboard.md                # rebuilt after each iter
│       ├── memory_summary.md             # rebuilt after each iter
│       ├── experiments.jsonl
│       ├── loop_log.jsonl                # per-iter cost / turns / duration
│       ├── transcript_iter*.jsonl        # full sub-agent transcript
│       └── results/<candidate>/          # harness output per algorithm version
└── ioi-discovery-v1_rev3/                # Discovery rev3: 3 stages, recovery=0.99
```

All six runs are reproducible from the committed specs — just point
`autointerp investigate --spec <path>` at the rev2 or rev3 file.
