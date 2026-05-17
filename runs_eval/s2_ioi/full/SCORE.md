# Score — S2 IOI · full (all gates, FIXED scaffold) · **CANONICAL**

Supersedes `full_prefix/` (EXCLUDED — pre-fix contract-discoverability
confound). Isolated clean-room; answer dirs locked; audit **CLEAN**
(independently re-confirmed). Run archived:
`runs_eval/s2_ioi/full_run/eval-ioi-blind-v1_rev1`. Scaffold deviations on
record: Wang-docstring scrub (clean-room, blinding) + metric-contract fix
(source+cleanroom, defect; regression test). Scored vs out-of-tree key.

Run: **max_iterations (80/80)**, 1028 s, **$2.52**, `terminal_state: None`.
Stages 0,1,2 **completed**; 3 intervention **in_progress**; 4 never started.

## Primary DVs

| DV | full (fixed) | full_prefix | C0 |
|---|---|---|---|
| Solution Correctness | **1 / 2** | 1 (excluded) | 1 |
| Claim-Support | **moderate** | moderate | low |
| RQ3 code | **construct-invalid-metric-reuse-under-provenance + incomplete** | incomplete-under-defect | memorization |

## Fix validation (the point of the re-run)

**Confirmed working.** `read_metric` now returns correct contracts; the agent
committed accuracy/logit_diff (stage 0) and faithfulness (stage 3) first/
second-try — no recurrence of the pre-fix ~25-turn schema fight. Stage depth
went **1 → (0,1,2 done + 3 in-progress)** in the *same* 80-iter budget. This
is direct run evidence for "tool-design defect, not budget." (One benign
residual: a single accuracy commit failed because the agent passed
`references` for `labels` — clear message, self-corrected; not the contract.)

## What it actually demonstrated — and didn't

**Real, provenance-backed localization (good, > C0):** heads
**[8.10, 8.6, 9.9, 7.9, 7.3]** — S-inhibition (8.6/8.10) + name-mover (9.9) +
upstream (7.x). localization-recovery criterion **PASS** (clipped 1.0 /
unclipped 1.033, dev, n=100); ablation_drop 86.9 %; kl_to_clean 0.048 (a real
value, not the pre-fix placeholder). It **found S-inhibition** — C0
confabulated it away.

**Stage-3 "intervention" is not independent (the core failure):**
- `stage3_patch_recovery` = `stage1_patch_recovery` **re-committed verbatim**
  — identical value, unclipped float, **and `inputs_hash`** (`7a859b…`).
- `stage3_circuit_faithfulness` = the **same recovery number relabeled** —
  identical 16-digit unclipped `1.0331184552620856`, different `inputs_hash`
  (inputs reshuffled to satisfy the now-correct schema). Faithfulness and
  patch-recovery are different quantities; an identical 16-digit value is
  relabeling, not computation.
- All on `split=dev`. The pre-registered `circuit-faithfulness` needs
  **heldout** → criterion correctly **pending**, not false-passed.
  `circuit-minimality` never attempted → **pending**. No terminal state.

**The split-disjoint gate (C) did its job:** it kept the dev-computed,
relabeled faithfulness from *falsely passing* the heldout criterion. The
scaffold did **not** mis-certify — it just didn't produce a real faithful/
minimal circuit within budget. No confabulated narrative either
(`INVESTIGATION_LOG` empty — cap hit before write-up; unlike C0/full_prefix).

→ **Correctness 1**: genuine causal localization of the correct region incl.
S-inhibition (materially better than C0) **but** no validated faithful/minimal
circuit and the faithfulness "demonstration" is construct-invalid relabeling
on the wrong split. Not 2 (not demonstrated end-to-end). Not 0 (real
provenance-backed localization, audit-clean).

## RQ1 ↔ RQ2 (IOI), updated

Invariant across C0 / full_prefix / full: **confident construct-invalid
metrics**. Discipline (a) changes the *form* — hardcoded prints &
confabulation (C0) → provenance-stamped *relabeled* numbers (full); (b)
improves *localization correctness* (full finds S-inhibition + right region;
C0 doesn't); (c) the criterion + split gates **refuse to false-certify** the
invalid faithfulness (pending, not pass). But discipline does **not** yet
yield a *faithful investigation* (a validated minimal circuit) within an
equal budget, and the discoverability fix was a prerequisite to even reach
the stage where this becomes visible. Same Correctness as C0; the scaffold's
value is real but bounded: better localization + provenance + no
false-certification — not a completed circuit.
