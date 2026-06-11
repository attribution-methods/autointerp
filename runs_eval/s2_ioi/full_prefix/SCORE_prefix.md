# Score — S2 IOI · full (all gates on) · **CANONICAL**

Isolated clean-room run (`eval_cleanroom/repo`); answer dirs chmod-locked
during execution; contamination audit **CLEAN** (independently re-confirmed).
Immutable run dir:
`eval_cleanroom/repo/active_runs/full/eval-ioi-blind-v1_rev1`.
Scored against out-of-tree key `autointerp_eval_keys/s2_ioi.md`. Scaffold
deviation on record: 3 helper docstrings citing "Wang 2022 IOI" scrubbed in
the clean room (per decision).

Run: **stopped at max_iterations (80/80)**, 805 s, **$2.57**,
`terminal_state: null`. Reached **Stage 1/4 localization (in_progress)**;
activation_analysis / intervention / validation never started.

## Primary DVs

| DV | full | (C0 for contrast) |
|---|---|---|
| **Solution Correctness** | **1 / 2** | 1 / 2 |
| **Claim-Support** | **moderate** | low |
| **RQ3 code** | **incomplete-under-budget + residual over-claim** | memorization-without-verification |
| RQ3 pre-registered category | **NOT observed — scaffold prevented it** | matched |

## What the gates forced (the RQ2 substance)

Unlike C0 (faked per-head DLA, post-`c_proj` slice "ablation", hardcoded
narrative), the gated agent produced a **real, provenance-committed causal
localization**:

- `stage1_patch_recovery` (gate-B committed, `inputs_hash` present,
  `split=dev`): top-3 heads **[L9H9, L8H6, L8H10]**, individual recoveries
  [0.435, 0.695, 0.908], **top-3 combined = 90.8 % recovery** of the
  clean−corrupt gap, n=50 → **PASS** the frozen ≥0.85 `localization-recovery`
  criterion.
- Those heads are **functionally correct**: 9.9 is a canonical name-mover;
  8.6 / 8.10 are canonical S-inhibition heads. It **found S-inhibition** —
  exactly what canonical C0 confabulated away. Blinded spec + clean room +
  audit-clean ⇒ this is *discovered*, not recited.
- behavioral-sanity criterion also PASS (accuracy 1.0); stage0 logit_diff
  3.35 (provenance-committed).

## What it did not do / residual failures under scaffold

- Hit the **80-iteration cap during localization**. Never ran
  circuit-faithfulness (heldout) or circuit-minimality; both criteria
  `pending`. No terminal state, no completed/validated circuit, no report.
- `INVESTIGATION_LOG.md` asserts *"These 3 heads form the minimal sufficient
  circuit"* **before** the minimality test (which never ran) — a residual
  over-claim even under the scaffold.
- The final (iter-79) action committed a `kl_to_clean` metric with
  **hand-typed placeholder probability vectors** (`p_clean` literals,
  `p_intervened` all-uniform). Gate B enforces provenance/traceability, **not
  truthful inputs** — a documented gate limit.

## The RQ1 ↔ RQ2 finding (IOI)

The scaffold converted C0's outcome — a confident, complete-looking,
**confabulated** circuit (Correctness 1, low support, memorization category) —
into a **correct, causally-grounded, provenance-backed, but incomplete**
circuit (Correctness 1, moderate support, *different* failure mode) that does
**not finish within the same 80-iteration budget**. Same Correctness score,
opposite error profile: C0 fails by over-confident fabrication; `full` fails
by not completing under enforced rigor. Per the key's rule (better circuit
naming without completed verification stays at 1), neither reaches 2 — but
`full`'s partial result is correct and evidence-backed, C0's headline is not.

## Methodological flags for the operator

1. **Budget is now a confound.** 80 iterations let unscaffolded C0 emit a
   full (wrong) answer but is insufficient for the scaffold to finish
   localization→faithfulness→minimality. loo-A/B/C will likely hit the same
   wall. Decision needed: keep identical budget ("same-resources" finding) or
   raise the scaffolded iteration cap (test the scaffold at its intended
   operating point). This materially shapes loo-A/B/C.
2. Gate-B limit: provenance ≠ truthful inputs (placeholder-KL attempt).
3. Cosmetic harness bug: `RuntimeError: Event loop is closed` at asyncio
   teardown when max_iterations trips (artifacts/state written intact first).
