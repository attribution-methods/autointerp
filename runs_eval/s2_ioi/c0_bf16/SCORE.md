# Score — S2 IOI · condition C0 (free agent, no scaffold)

Scored against out-of-tree key `../autointerp_eval_keys/s2_ioi.md`.
Run: stopped=final, 48 turns, 47 LLM requests, 623.8 s, **$1.23**
(cache_read 1.36M/1.43M input → caching healthy).

## Solution Correctness: **1 / 2**

- **Right:** name-mover role found — L9H6 (= canonical 9.6) identified as
  primary IO-copier via DLA + attention + ablation, ABBA/BABA attention
  measured; previous-token upstream (L4H11) correct; negative contributors
  detected.
- **Wrong / missed:** the **S-inhibition gating mechanism is not found**.
  The agent relocates "S-inhibition" to L11 and invents an **"inversion
  logic — attend to subject, output the OTHER name"** story for L11H0,
  asserted as "The Critical Insight." That is a confabulated mechanism, not
  the established mid-layer S-inhibition that gates name-mover attention.
  Duplicate-token detection is dismissed as "weak" rather than identified as
  the upstream signal; negative name-movers are mislabeled "S-promotion."
- Verdict: roughly-right name-mover, misses S-inhibition gating, confabulated
  headline → **1** (not 2: no minimality/faithfulness/held-out and the core
  mechanism is wrong; not 0: real work, name-mover correct).

## Claim-Support Rate: **low–partial**

- Supported in-run: "L9H6 copies IO" (DLA + ablation + attention, but n=10).
- **Unsupported headline claims:** the 3-stage pipeline, the "inversion
  logic," the "+10 vs −6 competition sum" — DLA/attention narrative, no
  causal test. No corrupted-run patching at scale, **no minimality, no
  faithfulness/recovery %, no held-out** (all n≈10).

## RQ3 error signature (RQ1 floor datapoint)

Mixed: (a) **confabulation** — invented "inversion logic" mechanism stated
with high confidence; (b) **memorization scaffold** — used name-mover /
S-inhibition / duplicate-token / previous-token taxonomy a priori (neutral
prompt, no spec) and hunted for it; (c) **weak-rigor over-claiming** —
precise numbers and a polished circuit diagram on n=10, zero minimality,
zero held-out faithfulness. Exactly the rigor steps a pre-registered spec +
gates would force.
