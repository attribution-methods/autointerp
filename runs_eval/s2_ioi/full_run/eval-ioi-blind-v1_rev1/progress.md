# Investigation progress — eval-ioi-blind-v1_rev1

Status: criterion_failed
Stage: 2/5 — localization
Updated: 2026-05-17T18:39:31.794379Z

## Stages
- [x] 0 black_box — completed (2 artifacts)
- [>] 1 localization — in_progress (2 artifacts)
- [ ] 2 activation_analysis — pending
- [ ] 3 intervention — pending
- [ ] 4 validation — pending

## Criteria
- PASS         behavioral-sanity — accuracy >= 0.95 (observed 0.998)
- FAIL         localization-recovery — patch_effect_recovery >= 0.85 (observed 0.4634269177913666)
- pending      circuit-generalization — The candidate circuit recovers at least 0.85 of the clean-corrupt logit_diff gap on the heldout split (disjoint name pairs not used during localization)
- pending      circuit-minimality — Every head in the final circuit causes a recovery drop >= 0.05 when removed on heldout

## Budget
tool_calls=10 bash_calls=0 gpu_seconds=0.0 wallclock=0.0s samples=0
