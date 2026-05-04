# Autointerp Agent — End-to-End on IOI

**TL;DR.** Sonnet-4.5 ran the agent **blind** on a pre-registered IOI spec
and recovered the canonical Wang-et-al-2022 circuit: dominant name-mover
**L9H9**, plus **L9H6** (correctly flagged as inhibitory) and **L10H0**.
All four pre-registered success criteria passed on a heldout split with
disjoint name pairs — heldout faithfulness **0.913**, minimality **0.164**.
Wallclock **~29 min**, 167 tool calls. Run dir
`runs/ioi-gpt2-small-blind-v1_rev1/`.

---

## What was added on top of the existing scaffold

Building on Uzay's structure, three pieces:

### 1. A `metrics/` reference layer

Each metric the agent is allowed to invoke is described as a Markdown card,
mirroring how skills work for tools. The agent looks them up via
`list_metrics` / `read_metric`, and the metric registry refuses anything
not on the list — so all metric values committed to a finding have a
deterministic, audited implementation. Example:

```markdown
# logit_diff
**Family:** behavioral · **Range:** unbounded · **Direction:** higher is better

`logit(target_token) - logit(foil_token)` at the prediction position.
Standard behavioral signal for contrastive tasks (IOI, factual recall,
role assignment).

## When to use
- You have a clean target/foil token pair per prompt.
- You want a continuous signal rather than discrete accuracy.

## Pitfalls
- Magnitude depends on model temperature scale; don't compare across
  models without normalizing.
- Mean across prompts can hide a heavy tail — also report per-prompt
  distribution.
```

The agent picks metrics during spec design (stage 0); the runtime then
enforces that pre-registered list during the investigation.

### 2. Stage 0 — pre-registered `InvestigationSpec`

A new conversational mode where the agent and a human together build a
falsifiable spec. Fields include `question`, `hypothesis`, `model`,
`dataset`, `contrast`, `stages`, and **`success_criteria`** wired to the
metric registry. Once the human approves and the spec is written to
`outputs/specs/<id>_rev<n>.json`, it is **immutable for the rest of the
run** — we don't trust the agent to re-litigate its own success criteria.

If results come back negative or surprising, the agent emits a revision
request and a *new* rev is written; the prior run dir is preserved.

The IOI spec we ran:
```
question: How does GPT-2-small mechanistically implement IOI?
hypothesis: Attention-head circuit at END copies the IO embedding...
stages:    black_box → localization → activation_analysis → intervention → validation
criteria:  accuracy ≥ 0.95
           patch_effect_recovery ≥ 0.85
           faithfulness ≥ 0.85   (heldout, disjoint name pairs)
           minimality ≥ 0.05     (per-component drop)
```

### 3. Investigation pipeline — gated tool surface

A coding agent walks `spec.stages` in order with two tiers of tools:

- **Tier 1 (free-form)**: `bash`, `read_file`, `write_file`, `edit_file`.
  This is where actual ML happens — the agent writes Python scripts,
  caches activations, runs forward passes.
- **Tier 2 (gated)**: `compute_metric`, `commit_artifact`,
  `evaluate_criterion`, `advance_stage`, `request_spec_revision`,
  `current_stage`, `get_state`, `get_budget`. These are the only tools
  that can move the run forward or write to `findings/`. They reject
  anything that doesn't match the pre-registered spec.

This makes p-hacking mechanically harder: the agent can't invent a metric
mid-run, can't change a threshold, and can't quietly drop a stage.

### 4. Granular patching utilities (added this session)

The first end-to-end run revealed the toolkit only had whole-layer /
residual patching, which on GPT-2-small recovers ~100% of the gap on
*any* layer (uninformatively flat). We added
`autointerp.tools.head_patching`:

| Tool | What it does |
| --- | --- |
| `cache_head_z` | Captures per-head pre-`W_O` activations `z` shape `[B, S, H, d_head]` |
| `mean_head_z` | Builds the standard mean-ablation baseline |
| `run_with_head_patches` | Forward pass with arbitrary `(layer, head, position)` overrides |
| `mean_ablate_heads` | Mean-ablate one or more sites |
| `head_patch_sweep` | `(L, H)` sweep, returns `[L, H]` recovery matrix |
| `path_patch` | Single-step path patching (sender clean, others frozen to corrupt) |
| `logit_diff` | Batched paired-token metric |

Family-agnostic — hooks `c_proj` (GPT-2) or `o_proj` (Llama / Qwen /
Gemma / Mistral) input.

---

## What the agent actually did on IOI

### Stage 0 — black-box behavioral baseline
Wrote and ran an IOI dataset generator (500 dev + 100 heldout, ABBA/BABA
balanced, single-token names, length-aligned for clean/corrupt pairs).
- `accuracy = 0.986` (≥ 0.95 ✓)
- `mean logit_diff = 3.3`

### Stage 1 — localization
Logit lens across all 12 layers. Then patched **all 144 (layer, head)
pairs** using `head_patch_sweep` at the END position.
- Top-3 heads `[(L9, H9), (L9, H6), (L10, H0)]` → simultaneous
  `patch_effect_recovery = 0.895` (≥ 0.85 ✓)
- L9H9 alone recovers **0.527** of the gap.

### Stage 2 — activation analysis
Mean-ablated each top head individually.
- **L9H9**: ablation drop `0.304`
- **L9H6**: ablation drop **`-0.286`** — agent's own note:
  *"Negative drop = ablating it improves performance. Suggests an
  inhibitory or backup role."*
- **L10H0**: ablation drop `0.247`

This matches the literature's "negative name mover" finding without any
hint in the spec or skills (we audited skill files and stripped IOI /
"name mover" mentions before the run).

### Stage 3 — path patching
Direct paths from each candidate head to the output:
- L9H9 → output: `0.574` recovery
- L9H6 → output: `0.202`
- L10H0 → output: `0.156`

Top-3 circuit faithfulness on dev: **0.895**.

### Stage 4 — heldout validation (disjoint name pairs)
- Faithfulness **0.913** (≥ 0.85 ✓)
- Minimality **0.164** (≥ 0.05 ✓) — every head's removal drops
  faithfulness by a non-trivial amount.

The circuit *generalizes better* on heldout than on dev (0.913 vs 0.895).

---

## Costs & footprint

| | |
| --- | --- |
| Model | `claude-sonnet-4-5` |
| Iterations | 97 (after a first run hit a 60-iter default cap and was resumed; cap is now 500 by default) |
| Tool calls | 167 |
| Wallclock | **~29 min** (00:43:59Z → 01:12:54Z) |
| GPU | 1× H100 (or whatever box you launch on); GPT-2-small is tiny |
| Disk | ~tens of MB per run dir (transcripts + activations cache + scripts) |

Token-level $ cost wasn't recorded in the run logs; can wire that up via
the SDK usage callbacks when we want it. Order-of-magnitude this run
was cheap (<<$10).

---

## What you can do with this right now

```bash
# 1. Build a spec interactively (Stage 0)
PYTHONPATH=src python -m autointerp_agent.spec --model anthropic/claude-sonnet-4-5

# 2. Run the investigation against an approved spec
PYTHONPATH=src python -m autointerp_agent.investigation \
  --spec outputs/specs/<id>_rev<n>.json \
  --model anthropic/claude-sonnet-4-5 \
  --auto-approve --max-iterations 500
```

The two CLIs are still separate — see open questions below.

Everything for the IOI run is in `runs/ioi-gpt2-small-blind-v1_rev1/`:
- `INVESTIGATION_LOG.md` — the agent's running narrative
- `assistant_turns.jsonl` / `tool_invocations.jsonl` — full debugging
  transcript (every tool call, args, outputs in `tool_invocations/`)
- `findings/stage_*/` — committed metric results
- `report.json` — final pre-registered evaluation
- `scripts/` — every Python script the agent wrote

---

## Open questions / next steps

1. **Orchestration.** Stage-0 spec design and the investigation are still
   two CLIs. Want a single orchestrator that hands the approved spec
   directly to the investigation pipeline (and optionally re-engages the
   user only when revision is needed).
2. **Revision loop.** When results come back negative or off-hypothesis,
   the agent should auto-draft a revised spec (new rev, prior run
   preserved) and propose it. Today this is manual.
3. **More test cases.** IOI is the easiest possible target. Next up:
   factual-recall localization, induction-head detection on small
   models, refusal/jailbreak circuits on instruct models. Each new
   target tests a different skill bundle.
4. **Caching across runs.** Right now activation caches are per-run;
   should be content-addressed and shared across reruns of the same
   model + dataset.
5. **Looser constraints experiment.** How much does the gated tool
   surface actually buy us in practice? Worth running a no-gating ablation
   on the same spec to quantify p-hacking risk.
6. **Cost telemetry.** Wire token usage into the runtime so per-stage $
   costs land in `report.json`.
7. **Bigger models.** GPT-2-small is small enough that even crude
   methods work. Re-run on Llama-3.1-8B or Qwen-2.5-7B to stress-test
   head patching at scale.

If folks want to dive in, easy ways to help:

- Add more **metric cards** under `metrics/` (e.g. attention-pattern
  metrics, OV-projection metrics).
- Add more **skill cards** under `skills/` (we're light on
  cross-attention and per-position interventions).
- Build the **orchestrator** above (1).
- Pick a **new target behavior** and write its spec; run it.
— happy to divide and iterate.
