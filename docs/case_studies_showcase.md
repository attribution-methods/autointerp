# Golden Case Studies — Showcase

A first-pass snapshot of how the autointerp agent does on the 10 golden questions
(`case_studies/golden.yaml`), run end-to-end on our current testbed:

- **driver model:** gpt-5-nano (a deliberately weak planner/agent)
- **target model:** a small open-weights model (GPT-2 / GPT-2-medium)

These are honest, current results — not cherry-picked. The point is to show the
agent's *reach* today, warts and all.

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
