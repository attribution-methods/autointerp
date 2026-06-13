# Investigation specs (case-study starting points)

Three approved `InvestigationSpec`s, reusable as ready starting points for the
case studies in [`../../docs/case_studies.md`](../../docs/case_studies.md):

| File | Stimulus | Target model | Case study |
|------|----------|--------------|------------|
| `s1_shutdown.json` | Shutdown resistance | `Qwen/Qwen2.5-7B-Instruct` *(confirm)* | #9 |
| `s2_ioi.json` | Indirect object identification | `gpt2` | #6 |
| `s3_surprise.json` | Surprise representation | `EleutherAI/pythia-1.4b` *(confirm)* | #4 |

`<name>.question.md` is the matched free-agent prompt
(`spec.question + "\n\n" + spec.behavior.description`) — the exact text to hand an
unscaffolded agent so the task is identical to a scaffolded run.

## Blinding (contamination control)

Each spec is written so the agent cannot read the answer off it: a truthful
high-level `question`; a `hypothesis` that names the *method space*, not the
result; stage `notes` that say "sweep / don't shortlist"; and `metadata` that
documents the redaction. Ground-truth answers and scoring rubrics live **out of
tree** — `docs/case_studies.md` holds them and must stay out of any agent run
context (run dir, clean room, prompt, or `bash`-reachable path).

## Running

```bash
.venv/bin/python -m autointerp_agent.investigation \
  --spec eval/specs/s3_surprise.json --auto-approve
```

`--ablation` defaults to `full` (all discipline gates on). Run dirs land under
`runs/<spec_id>_rev<n>/` (gitignored). See
[`../../docs/investigation.md`](../../docs/investigation.md) for the pipeline,
gate APIs, and run-directory layout.
