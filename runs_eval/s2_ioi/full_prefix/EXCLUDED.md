# EXCLUDED — pre-fix (metric-contract discoverability defect)

This `full` run is confounded by the scaffold defect where advertised metric
contracts (`spec.METRIC_META.requires_inputs`, surfaced by `read_metric`)
disagreed with enforcement (`metrics.py _require_keys`) for 6/15 commit-
validated metrics — incl. logit_diff/kl_to_clean/faithfulness/minimality on
the IOI critical path. >50% of the 80-iter budget was burned reverse-
engineering the wrong contract, so the "incomplete / Correctness 1" outcome is
a defect artifact, not a discipline signal (bf16-class confound).

Fixed at source (registry reconciled to enforcement; regression test
`tests/test_metric_contract_consistency.py`). Re-run `full` on the fixed
scaffold is the canonical datapoint. Kept only as evidence of the as-shipped
defect (itself an RQ finding). Not scored, not in RESULTS.
