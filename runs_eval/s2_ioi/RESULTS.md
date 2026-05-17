# S2 IOI — cross-condition results

Meta-eval of the autointerp scaffold on the IOI stimulus. Each condition is
scored against the out-of-tree key `autointerp_eval_keys/s2_ioi.md` with the
**same schema** (`<condition>/score.json` + `<condition>/SCORE.md`). "Faithful"
= reached the real solution with in-run causal evidence, not circuit-name
overlap.

- Environment held constant across all 5: GPT-2-small **float32**, model
  `openrouter/anthropic/claude-sonnet-4.5`, max_iter 80, wallclock 1800 s.
- **C0** = free agent, no scaffold (RQ1 floor). **full** = all gates on
  (byte-identical original scaffold). **loo-A/B/C** = leave-one-out:
  A = frozen-spec off, B = provenance/canonical-metrics off,
  C = split-disjointness off.
- Excluded: `c0_bf16/` (pre-fix bf16 run; kept only as precision-bug
  evidence — never scored, never in this table).

## Primary DVs

| Condition | Correctness /2 | Claim-Support | RQ3 code | $ | reqs |
|---|---|---|---|---|---|
| **C0** (no scaffold) | **1** | low | memorization-without-verification | 0.87 | 34 |
| full (all gates, **fixed scaffold**) | **1** | moderate | construct-invalid-metric-reuse + incomplete | 2.52 | 80\* |
| ~~full (pre-fix)~~ | EXCLUDED | — | contract-discoverability confound | 2.57 | 80\* |
| loo-A (no frozen spec) | _TBD_ | _TBD_ | _TBD_ | | |
| loo-B (no provenance) | _TBD_ | _TBD_ | _TBD_ | | |
| loo-C (no split-disjoint) | _TBD_ | _TBD_ | _TBD_ | | |

## Mechanism / process matrix

The RQ2 question: **which gates force the verification C0 skipped, and does
that move Correctness off 1?** Score each cell Y/N/n-a from the run artifacts.

| Check (what discipline should force) | C0 | full | loo-A | loo-B | loo-C |
|---|---|---|---|---|---|
| Pre-registered hypothesis honored (not retrofitted) | n/a | **Y** | _TBD_ | _TBD_ | _TBD_ |
| Correct functional roles named (nm + S-inhib + dup upstream) | Y | **Y** (S-inhib found) | _TBD_ | _TBD_ | _TBD_ |
| Per-head/component DLA actually computed (not faked) | **N** | n/a (patching) | _TBD_ | _TBD_ | _TBD_ |
| Ablation method valid (true head isolation) | **N** | **Y** | _TBD_ | _TBD_ | _TBD_ |
| Causal patching done & differentiated | Y | **Y** (dev, n=100) | _TBD_ | _TBD_ | _TBD_ |
| Minimality tested | **N** | **N** (pending) | _TBD_ | _TBD_ | _TBD_ |
| Faithfulness / held-out recovery reported | **N** | **N** (relabeled, dev≠heldout) | _TBD_ | _TBD_ | _TBD_ |
| Every metric tied to a committed artifact (gate B) | **N** | **Y** (but construct-invalid) | _TBD_ | _TBD_ | _TBD_ |
| No hardcoded-narrative / unconditional "confirmed" | **N** | **Y\*** (no narrative; metrics relabeled instead) | _TBD_ | _TBD_ | _TBD_ |
| n causal clean/corrupt pairs | 1 | **100** (dev) | _TBD_ | _TBD_ | _TBD_ |

> `full`/loo columns reset: the first `full` run is **EXCLUDED** (pre-fix,
> defect-confounded — see "Pre-fix `full` + the discoverability defect"
> below). All scaffolded conditions will be (re-)run on the fixed scaffold.

## C0 summary (canonical, fp32)

Names the true structure (duplicate-token → S-inhibition → name-mover) in the
right order, but: headline "+110.4 L11 attn" is a through-unembedding
artifact; per-head DLA is hardcoded print; head "ablation" zeroes a
post-`c_proj` slice (not a head) with several signs contradicting the
assigned roles; only one sound causal experiment (component activation-patch:
L9attn 64.4% / L11attn 63.7% recovery, n=1, no minimality). → Correctness 1,
Claim-Support low, RQ3 = the pre-registered memorization-without-verification
category. Full per-claim ledger in `c0/SCORE.md`.

Stability finding (RQ1): bf16→fp32 (precision only, identical prompt) changed
the circuit *content* but not the *failure mode* — bf16 not reproducible,
fp32 canonical.

## Pre-fix `full` + the discoverability defect

The first `full` run (isolated, audit-clean) hit the 80-iter cap mid-
localization and produced no completed circuit. Turn-by-turn analysis showed
**>50 % of the budget was burned fighting an undiscoverable metric contract,
not doing interpretability.** Root cause (primary-sourced): the scaffold's
*advertised* metric contract — `spec.METRIC_META[*].requires_inputs`, surfaced
verbatim by the `read_metric` tool — **disagreed with what
`compute_and_commit_metric` enforces** (`metrics.py _require_keys`) for **6 of
15 commit-validated metrics**, incl. `logit_diff`, `kl_to_clean`,
`faithfulness`, `minimality` — all on the IOI critical path. `read_metric`
didn't fail to state the contract; it stated the **wrong** contract, so the
agent could only learn it by failing the gate repeatedly.

This is a **bf16-class confound** (an incidental implementation defect that
swamps the construct under study), not a discipline signal. ⇒ the pre-fix
`full` run is **EXCLUDED** (`full_prefix/`, `SCORE_prefix.md`); the "budget
confound" framing is **withdrawn** — it was a tool-discoverability defect, not
a budget problem.

**Fix (gate-invariant):** registry `requires_inputs` reconciled to the
authoritative enforced `_require_keys` (enforcement/validation byte-
unchanged); regression test `tests/test_metric_contract_consistency.py` pins
registry==enforcement. Applied at source + re-seeded clean room. The
as-shipped defect is itself reported as an RQ finding (the scaffold ships with
its documented metric contract inconsistent with enforcement → high, avoidable
compliance overhead — directly relevant to "minimal viable discipline").

### Recorded scaffold modifications (deviations from as-shipped)

1. **Blinding (clean-room only):** 3 helper docstrings citing "Wang 2022 IOI"
   scrubbed (`spec.py:179`, `metrics.py:117`, `head_patching.py:384`).
2. **Defect fix (source + clean room):** 6 `METRIC_META.requires_inputs`
   corrected to match enforcement + regression test. No gate semantics change.

All scaffolded conditions (full, loo-A/B/C) run on this fixed scaffold; the
single pre-fix `full` is the only excluded scaffolded datapoint.

## How to score a new condition

1. Scaffolded runs: `eval/run_isolated.py --only <cond>` (clean-room, answer
   dirs locked, audit gate); the audited run is archived to
   `runs_eval/s2_ioi/<cond>_run/`. C0 is the fp32 `c0/` run. Score from there.
2. Read `c0_answer.md`/report + open the scripts behind each causal claim;
   record claim → script → what it computed (no verdict) as in `c0/SCORE.md`.
3. Fill `<cond>/score.json` (same keys as `c0/score.json`) and `<cond>/SCORE.md`.
4. Update the two tables above. The decisive comparison is the mechanism
   matrix: a gate "works" for RQ2 only if it flips an **N** to **Y** *and*
   that raises Correctness — naming the circuit better without real
   verification stays at 1 (the key's explicit rule).
