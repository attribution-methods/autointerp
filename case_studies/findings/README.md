# Findings — what autointerp currently finds

One file per case (`<case_id>.md`) recording the **agent's own output** for that
question on our current testbed (gpt-5-nano driver + a small open-weights model
such as GPT-2 / GPT-2-medium). These are safe to commit: they are what the agent
produced, and contain **no ground truth** (the agent never had access to it).

Each file notes the model used, the plan/metric the agent designed, its verdict
(PASS / FAIL / INCONCLUSIVE / could-not-attempt), and a short takeaway. Most of
the research-grade questions need larger or instruct-tuned models than our
testbed provides, so an honest "not tractable on a small model yet" is itself a
finding about the agent's current reach.

The detailed scoring of these findings against the private ground truth lives in
the private store, not here. A high-level public comparison is in
`docs/case_studies_showcase.md`.
