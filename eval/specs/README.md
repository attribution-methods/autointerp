# Eval stimuli (scaffold-faithfulness meta-eval)

Three blinded, approved `InvestigationSpec`s — the stimuli for
`docs/scaffold_faithfulness_eval.md`.

| File | spec_id | Stimulus | Target model | Error category (RQ3) |
|------|---------|----------|--------------|----------------------|
| `s1_shutdown.json` | `eval-shutdown-blind-v1` | S1 Shutdown Resistance | `Qwen/Qwen2.5-7B-Instruct` *(default — confirm)* | motivated/anthropomorphic confabulation |
| `s2_ioi.json` | `eval-ioi-blind-v1` | S2 IOI | `gpt2` | memorization-without-verification |
| `s3_surprise.json` | `eval-surprise-blind-v1` | S3 Surprise | `EleutherAI/pythia-1.4b` *(default — confirm)* | artifact-conflation |

`<name>.question.md` is the **matched C0 prompt**, generated verbatim as
`spec.question + "\n\n" + spec.behavior.description`. The task-5 driver feeds
this exact text to the free-agent harness so the task is provably identical
to the scaffolded conditions (only discipline differs).

## Blinding

Each spec follows the `examples/specs/ioi-gpt2-small-blind-v1_rev1.json`
convention: truthful high-level `question`; `hypothesis` states the answer is
"unknown a priori, discover empirically" and names the *method space* (not
the answer); stage `notes` say "sweep / don't shortlist"; `metadata`
documents the redaction and the pre-registered error category. The agent
cannot read the answer (or, for S1, an intent narrative) off the spec.

## Scoring keys are OUT OF TREE

Ground-truth + rubrics live **outside the repo** at
`../autointerp_eval_keys/{s1_shutdown,s2_ioi,s3_surprise}.md` (i.e.
`/lambda/nfs/test-filesystem/fadi/attribution-methods/autointerp_eval_keys/`).
They are never placed in an agent context, spec, prompt, or `runs/` dir.
Residual risk (documented): the agent has read-only `bash` and could in
principle explore the filesystem; the keys are a sibling of the repo, not
referenced anywhere, which defeats incidental leakage but not deliberate
hunting (not the failure mode under study).

## Running (task 5)

Run dir = `runs_root/<spec_id>_rev<n>` (independent of ablation flags), so
each condition needs its **own `--runs-root`**:

```
python -m autointerp_agent.investigation --spec eval/specs/s2_ioi.json \
  --ablation full   --runs-root runs_eval/s2/full   --auto-approve
python -m autointerp_agent.investigation --spec eval/specs/s2_ioi.json \
  --ablation A      --runs-root runs_eval/s2/loo-A  --auto-approve
# … B, C …
python -m autointerp_agent.free_agent --question-file eval/specs/s2_ioi.question.md \
  --run-dir runs_eval/s2/c0
```

All runs use `claude-sonnet-4-5`, `max_iterations=80`, the harness wallclock
kill, and `PYTHONPATH=src` (see the env-checkout-mismatch note).
