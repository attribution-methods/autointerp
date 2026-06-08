# Investigation progress — eval-ioi-blind-v1_rev1

Status: in_progress
Stage: 2/5 — localization
Updated: 2026-05-16T13:39:25.471423Z

## Stages
- [x] 0 black_box — completed (2 artifacts)
- [>] 1 localization — in_progress (1 artifacts)
- [ ] 2 activation_analysis — pending
- [ ] 3 intervention — pending
- [ ] 4 validation — pending

## Criteria
- PASS         behavioral-sanity — accuracy >= 0.95 (observed 1.0)
- PASS         localization-recovery — patch_effect_recovery >= 0.85 (observed 0.9075642413470237)
- pending      circuit-faithfulness — Candidate IOI circuit achieves faithfulness >= 0.85 on the heldout split
- pending      circuit-minimality — Every head in the final circuit causes a faithfulness drop >= 0.05 when removed (minimality)

## Budget
tool_calls=8 bash_calls=0 gpu_seconds=0.0 wallclock=0.0s samples=0
