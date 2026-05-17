# EXCLUDED — do not score, do not include in RESULTS.md

This is the pre-fix **bf16** C0 run. It is **not** the canonical C0 and is
**excluded from the scaffold-faithfulness comparison**. The canonical C0 is
`../c0/` (float32).

Kept only as evidence for the precision-bug finding: `load_model` defaulted
to `torch.bfloat16`, which 0.5-quantizes GPT-2 logit-diffs and silently
nulled this run's causal ablations (L11H0/L8H3 Δ=0.000) while DLA stayed
fine-grained. See `memory/project_bf16_logit_quantization.md` and the
"Stability note" in `../c0/SCORE.md`.

Safe to delete if the disk matters and the finding is no longer needed; it
plays no role in any score or aggregate.
