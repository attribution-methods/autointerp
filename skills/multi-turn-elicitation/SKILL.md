---
name: multi-turn-elicitation
description: Multi-turn auditor harnesses for behaviors that single-turn prompts cannot elicit. Use for agentic misalignment, refusal-under-pressure, sycophancy escalation, deception, scheming, alignment-faking, or any target behavior that requires sustained context, role-play setups, or graduated pressure across turns. The reference tool is Inspect Petri (auditor/target/judge tripod); this skill explains when to reach for it and how to plug its outputs back into the investigation pipeline.
---

# Multi-Turn Behavioral Elicitation

## When to use

Reach for a multi-turn auditor harness when the behavior under
investigation **cannot be reliably elicited from a single user message**.
Signs the behavior is multi-turn:

- The published methodology used a multi-turn setup (Anthropic's
  agentic-misalignment study, alignment-faking, the sycophancy ramp-up
  papers, refusal-under-pressure).
- The behavior depends on graduated pressure: the auditor needs to escalate
  across turns before the target capitulates.
- The behavior requires a *role/scenario commitment* that can't be packed
  into one turn (agentic agent given a goal, then progressively told the
  goal conflicts with safety).
- A single-turn `ContrastSpec` (positive/negative template with one swap)
  cannot isolate the behavior because the behavior depends on how the
  conversation accumulated.

When in doubt, run a 1-shot single-turn pilot first: if the behavior
elicits at the rate you need from a clean prompt, skip multi-turn.

## The auditor / target / judge tripod

Three roles, three different model choices:

| Role | Typical model | Job |
|---|---|---|
| **Auditor** | A capable assistant model (Claude Sonnet, GPT-class) | Drives the conversation. Given a `seed_instruction` describing the behavior to elicit, plans pressure tactics across N turns. |
| **Target** | The model under investigation (the one you're studying) | Responds in role. The auditor's messages are the target's *user* messages; the auditor never sees the target's chain-of-thought (just its responses). |
| **Judge** | A capable evaluator (Claude Haiku is plenty) | After the transcript is complete, scores whether the target's final response actually exhibits the behavior. Returns a label + a `tipping_turn` index. |

Why three roles, not one:
- The auditor must not also score, or it will declare success on its own
  bait. Separation prevents this confound.
- The judge must use a fresh LLM (not the target) so it isn't reading its
  own conversation back.
- Mock the auditor and judge during dev (deterministic canned transcripts)
  before paying for real elicitation.

## Inspect Petri (reference implementation)

[Inspect Petri](https://github.com/meridianlabs-ai/inspect_petri) is the
canonical open-source multi-turn auditor harness. It runs on Inspect AI;
auditor + target + judge are wired as Inspect agents.

A working install lives at `../circuitbreaker/.venv-petri/` (Python 3.12,
vendored petri at `../circuitbreaker/behavioral_stage/_vendor/inspect_petri/`).
The `behavioral_stage` package wraps it with caching, batch APIs, and
artifact persistence. **Don't reimplement either of these — invoke them
through `bash`.**

### Invocation pattern

```bash
# 1. Smoke test against the mock provider (no API key, no GPU).
cd ../circuitbreaker
python3 behavioral_stage/scripts/run_behavior.py \
  --behavior <BEHAVIOR_NAME> \
  --only-stage petri

# 2. Real elicitation with vLLM target + Anthropic auditor / judge.
#    Requires:
#      - vLLM server up (TARGET_PROVIDER=vllm, VLLM_BASE_URL set)
#      - ANTHROPIC_API_KEY in behavioral_stage/.env
#      - PETRI_FOLD_SYSTEM_INTO_USER=1 for Gemma (no system role in chat template)
PETRI_FOLD_SYSTEM_INTO_USER=1 LLM_PROVIDER=anthropic TARGET_PROVIDER=vllm \
  ../circuitbreaker/.venv-petri/bin/python \
  ../circuitbreaker/behavioral_stage/scripts/run_behavior.py \
  --behavior <BEHAVIOR_NAME>
```

Outputs land in `../circuitbreaker/behavioral_stage/outputs/<behavior>/`:

```
<transcript_id>/
  transcript.json          # auditor↔target turns + tipping_turn + judge label
  analysis.json            # mechanism distillation (Sonnet)
  prompts.json             # prefill-paired single-turn prompts
  completions/<id>.json    # target completions per prompt
pairs.jsonl                # behavior-wide accepted (clean, corrupt) pairs
```

### Plugging back into the autointerp pipeline

Two patterns, depending on what you need:

1. **Use the multi-turn transcripts directly as evidence.** Read
   `transcript.json`, attach the auditor↔target turns to a
   `BehavioralFinding` artifact (`commit_artifact`), score with a
   compute_metric like `hit_rate` over the judge labels.

2. **Use the prefill-paired single-turn prompts for activation analysis.**
   `pairs.jsonl` already contains (clean, corrupt) prompts that can be
   committed as a `PromptBatch` artifact. Each pair is balanced: same
   prompt, prefill commits the target to either behavioral or
   non-behavioral direction. Use these for activation patching, probe
   training, SAE feature lookup at the decision token.

The decision token is `pairs.jsonl[i].metadata.decision_position` — the
first generated token of the target's response. That's where the
contrast bites.

## Mock-first development

`behavioral_stage` ships canned transcripts under `LLM_PROVIDER=mock`.
Build out your investigation script against the mock provider until the
end-to-end flow works (artifact shapes, metric computation, criterion
evaluation), then flip to real providers for the paid run.

```bash
# Dev loop (no money, no GPU)
LLM_PROVIDER=mock TARGET_PROVIDER=mock \
  python3 ../circuitbreaker/behavioral_stage/scripts/run_behavior.py \
  --behavior <BEHAVIOR_NAME> --n-transcripts-per-seed 1
```

## Pitfalls

- **Auditor leakage.** If the auditor knows what behavior it's eliciting,
  it can hand the answer to the target via subtle phrasing
  ("you might consider…"). Use Sonnet-class auditors and abstract seed
  instructions; never give the auditor the rubric.
- **Judge contamination.** Don't reuse the auditor as the judge. The judge
  reads only the final response, not the auditor's plan.
- **Tipping-turn drift.** The "tipping turn" identified by the judge is
  one human's interpretation of when the target committed. Use it as a
  prior for where to attach probes; don't treat it as ground truth.
- **Open-weights replication.** Some published multi-turn behaviors don't
  replicate on smaller open models (the original studies often used
  frontier closed models). Check `hit_rate` early — if it's <10%, the
  target may not have the capability under investigation, and your spec
  needs a `request_spec_revision` exit.
- **Multi-turn → single-turn translation.** The agent typically wants
  single-turn prompts at scale (white-box analysis needs N≫1). Use the
  `analysis.py` mechanism distiller + `distiller.py` prefill expansion to
  go from 1 multi-turn transcript → N controlled single-turn prompts.
  These are downstream of the auditor.

## Related skills

- `prefill-pair-generation` — when contrast cannot be templated by
  swapping a single token (most behaviors).
- `causal-validation` — once you've localized to a layer/feature/direction
  using single-turn prompts derived from the multi-turn elicitation,
  validate causally on heldout transcripts the auditor produces with
  disjoint seeds.
- `pretrained-saes` — pair multi-turn elicitation with SAE feature
  lookup at the decision token for the cleanest "what fires before
  behavior X" reading.
