# Golden Case Studies — Showcase

A first-pass **baseline** snapshot of how the autointerp agent does on the 10
golden questions (`case_studies/golden.yaml`), run end-to-end on our current
testbed — followed by an [**after-optimization**](#after-the-optimization-pass)
section showing what a round of scaffold fixes (driven by these very cases)
changed:

- **driver model:** gpt-5-nano (a deliberately weak planner/agent)
- **target model:** a small open-weights model (GPT-2 / GPT-2-medium)

These are honest, un-cherry-picked results. The point is to show the agent's
*reach* — the baseline below, warts and all, and then the concrete before→after
once the bottlenecks it exposed were fixed.

> **No answers here.** This showcase reports only what the **agent** produced and
> how rigorous it was. The expected answers live in a separate private store and
> are never committed (the agent can read any file in its checkout, so a
> ground-truth answer in the repo would be reward-hackable — see
> [`case_studies.md`](case_studies.md)). The detailed scoring of each finding
> against the real answer is done privately. Per-case agent write-ups are in
> [`case_studies/findings/`](../case_studies/findings/).

## Scorecard

| # | Question (case_id) | Answer | Agent verdict | Did it actually answer it? |
|---|---|---|---|---|
| 1 | `emotions-before-misalignment` | unpublished | ✓ SUPPORTED | No — behavioral `logit_diff` proxy at `>= 0`; causal stage never ran |
| 2 | `subliminal-learning-mechanism` | unpublished | ⏸ paused | No — behavioral half ran; stalled producing causal patching evidence |
| 3 | `load-bearing-cot-tokens` | published | ⏸ paused | No — GPT-2 doesn't really do chain-of-thought; stalled before causal evidence |
| 4 | `surprise-representation` | published | ✗ NOT supported | No — fixed-token `logit_diff` proxy on 40 prompts; a non-detection |
| 5 | `training-free-activation-verbalization` | unpublished | ✓ SUPPORTED | No — no verbalizer built; rests on a logit-separability proxy |
| 6 | `indirect-object-identification` | published | ✗ NOT supported | No — behavioral accuracy on 4 prompts; localized no circuit components |
| 7 | `where-attention-concentrates` | published | ? INCONCLUSIVE | Partly — measured a signal; the causal/perturbation stage silently skipped |
| 8 | `entity-attribute-tracking` | published | ✓ SUPPORTED | No — black-box logit-margin proxy; localization/patching never ran |
| 9 | `shutdown-resistance` | published | ✗ NOT supported | No — a base model has no agentic behavior; rests on a 2-word logit comparison |
| 10 | `being-the-assistant-representation` | unpublished | ⏸ paused | No — base GPT-2 has no assistant persona; causal stage never ran |

Tally: **3 SUPPORTED, 3 NOT supported, 1 inconclusive, 3 paused** — and in **0 of 10**
did the agent recover the actual mechanism the question asks about.

## What this snapshot actually shows

**The pipeline works; the science doesn't land yet — on this testbed.** Across all
ten, the agent reliably did the *engineering*: it drafted a pre-registered plan,
generated data, ran a small open-weights model, computed a **grounded** metric
(traceable to a real forward pass), and reached a terminal verdict without
crashing or fabricating. That is exactly what the recent scaffold hardening was
for.

But in **every** case the substantive half — localizing a circuit, isolating a
direction, running a causal intervention — either never executed or stalled. So:

- The **SUPPORTED** verdicts are not "the agent answered the question." They are a
  weak planner picking a tractable behavioral proxy with a permissive threshold
  (e.g. `logit_diff >= 0`) and passing it. The findings say so plainly.
- The **paused** cases are the honest ones: the agent correctly recognized it
  couldn't produce the causal evidence on this testbed and stopped instead of
  bluffing.
- The two recurring bottlenecks are (a) the **model** (GPT-2-class can't exhibit
  chain-of-thought, agentic shutdown, or an assistant persona) and (b) the
  **causal-intervention tooling** stalling before patching evidence is produced.

The honest one-line summary: **autointerp currently demonstrates a robust,
grounded investigation *loop*, not yet correct mechanistic *answers*.** Recovering
the real results in this catalog will need stronger models (instruct/large) and a
more push-button causal-intervention path — both good next targets.

See [`case_studies/findings/`](../case_studies/findings/) for the full per-case
write-ups.

## After the optimization pass

This catalog was then used as the test bed for an agent-optimization round (same
weak `gpt-5-nano` driver, same local targets — the point was to fix the *scaffold*,
not to swap in a smarter agent). Re-running the cases surfaced concrete, general
bugs, each fixed at the root (no answer hardcoding). The fixes line up exactly with
the two bottlenecks named above:

**Bottleneck (b) — the causal-intervention / capture path stalled:**
- *Single-head patching reported `recovery ≈ 0` for distributed circuits* — the
  real bug behind a `patch_effect_recovery = 0.0`, not a true negative. `path_patch`
  now patches a **set** of heads and `circuit_recovery_capture` is one call
  (sweep → top-k → patch together → record). gpt2 IOI: single best name-mover head
  **0.12** vs the top-3 set **~0.97**.
- *The agent faked captures to pass the gate* — `record_capture(..., {"prompts":
  []}, source="model_forward")` stubs, then looped stub → fail → revise. An empty /
  placeholder `model_forward` capture is now **rejected at write time**, closing the
  fabrication hole the provenance layer always claimed to close.
- *`format_messages` crashed on base models* — it gated on
  `hasattr(tok, "apply_chat_template")` (true on gpt2, but no template is *set*), so
  every IOI generation came back `<error: chat_template not set>` and scored 0.0. Now
  gates on the template being set and falls back to plain completion.
- Causal captures are push-button (`autointerp.tools.causal_metrics`), surfaced in
  the prompt with the exact import; head-patching now works on GPTNeoX/Pythia too.

**Bottleneck (a) — the model didn't match the phenomenon:**
- The planner now frames prompts in the model's native format (next-token
  completion for base models; the standard IOI completion paradigm, not a Q&A quiz),
  uses an adequate sample size (tens, not a handful), and **routes
  reasoning/agentic/persona questions to an instruct/reasoning model from turn one**
  instead of defaulting to GPT-2.

### Before → after (cases actually re-run)

| # | case_id | Before | After |
|---|---|---|---|
| 6 | `indirect-object-identification` | ✗ NOT supported — Q&A quiz to gpt2, accuracy 0.25 on **4** prompts, no circuit | ✓ **SUPPORTED** — IOI completion task, `hit_rate = 1.0` on **200** real `model_forward` trials, zero chat-template errors |
| 4 | `surprise-representation` | ✗ NOT supported — fixed-token `logit_diff` proxy, a non-detection | ✓ **grounded SUPPORTED** — a discovered direction *causally* validated, `patch_effect_recovery = 0.43` on `gpt-neo-125M` |
| 3 | `load-bearing-cot-tokens` | ⏸ paused — planned on GPT-2, which emits no CoT | planner now finalizes **`falcon-7b-instruct`** (a model that *does* reason) from turn one, stating the question needs it |

**Honest scope (unchanged where it should be).** These are real improvements to the
*loop*: IOI went from a hard crash to a grounded, 200-trial verdict; surprise went
from a non-detection to a direction with a causal test. But IOI is still confirmed
**behaviorally**, not by localizing the name-mover circuit (the agent gated on
`hit_rate` rather than the localization metric), and the genuinely capability-bound
cases (emotions, subliminal, assistant persona) improve only at the *method* level —
the agent now picks the right model and the right causal test, but landing clean
mechanistic verdicts there still needs a stronger driver actually run end-to-end.
The scaffold bottlenecks are fixed; closing the science the rest of the way is the
next round.
