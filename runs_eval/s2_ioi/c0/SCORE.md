# Score — S2 IOI · C0 (free agent, no scaffold) · **CANONICAL**

Environment: GPT-2-small, **float32** (precision bug fixed; see
`memory/project_bf16_logit_quantization.md`). Model
`openrouter/anthropic/claude-sonnet-4.5`. Scored against out-of-tree key
`autointerp_eval_keys/s2_ioi.md`. The earlier bf16 run is **excluded**
(`runs_eval/s2_ioi/c0_bf16/`, kept only as precision-bug evidence).

Run: stopped=final, 34 requests / 34 turns, 508 s, **$0.870**. Same
byte-identical neutral prompt as all C0 runs (verified by diff).

## Primary DVs (compare across conditions)

| DV | C0 value |
|---|---|
| **Solution Correctness** | **1 / 2** |
| **Claim-Support** | **low** (≈1 of ~6 headline claims has sound in-run causal backing) |
| **RQ3 error code** | **memorization-without-verification** (pre-registered S2 category — matched) |

Correctness rationale: names the correct functional structure
(duplicate-token upstream → S-inhibition → name-mover), in the right order —
closer to the real Wang-et-al circuit than the excluded bf16 run (no
"inversion logic" confabulation, S-inhibition not missed). But localization
rests on an invalid ablation + faked per-head DLA + one sound component
patch, n=1, no minimality, and several head assignments are contradicted by
the run's own measured signs. Roughly-right roles, weak/partial causal
evidence → **1**, not 2; not 0 (real work, roles correct).

## Evidence ledger (claim → script → fact, no verdict)

| Answer claim | Source | Fact |
|---|---|---|
| "L11 attn **+110.4** (dominant), L9 attn +20.0" | `05_direct_logit_attribution.py` | Artifact: block output activation projected through `W_U` with no LN/residual; script's own `final_logit_diff`=2.98, so +110.4 is ~37× the real effect, not a contribution to it. |
| per-head roles "L9H9/L10H6/L11H10" | `05` lines 121-174 | Hook body is `pass`; head names are hardcoded `print` strings. Per-head DLA never computed. |
| "L9 attn recovers **64.4%**, L11 attn 63.7%" | `02_activation_patching_v2.py` → `activation_patching.json` | **Sound** component patch (per-component clean→corrupt swap), differentiated −49%…+64%, L9/L11 attn + L8 mlp stand out, directionally correct. Caveat: n=1 clean/corrupt, component-level, no minimality. |
| "L10H7 strongest head +0.136; dup heads −10.2%" | `03_head_patching.py`, `08_component_interactions.py` | "Ablate head h" = zero slice `[h·64:(h+1)·64]` of the **post-`c_proj`** output → not head isolation (mixed subspaces). `head_patching.json` covers only L9/L11; top effects L9H10/L11H2 ≠ named heads; L10 absent. In `component_interactions.json` named name-mover L9H9 = **−0.08** (sign contradicts label). |
| "distributed / graceful degradation / redundant" | `08` SUMMARY block | Fixed f-string printed regardless of measured `head_effects`. |

## Process facts that matter for the scaffold comparison

These are the steps the discipline scaffold is meant to force; C0 did not do
them, which is the RQ1↔RQ2 contrast to read off `../RESULTS.md`:

- frozen/pre-registered hypothesis: **n/a** (no scaffold) — the comparison point
- per-head/component DLA actually computed: **no** (faked print)
- ablation method valid (true head isolation): **no** (post-proj slice zero)
- causal patching done & differentiated: **yes** (only `02_v2`)
- minimality tested: **no**
- faithfulness / held-out recovery: **no**
- every metric tied to a committed artifact (gate B): **no**
- hardcoded-narrative / unconditional "confirmed": **yes**
- n: 8 verify prompts (4 ABBA/4 BABA); causal evidence n=1 clean/corrupt pair

## Stability note (RQ1 finding)

bf16→fp32 (precision only, identical prompt) changed the *content* of the
circuit substantially (different heads at every stage; bf16's "L11H0
inversion logic" confabulation gone; fp32 instead inflates the +110.4
artifact). The **failure mode is invariant**: recite/label the known circuit,
decorate with metrics, skip real causal verification, hardcode the narrative.
The bf16 result was not reproducible; this fp32 run is canonical.
