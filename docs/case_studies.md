# Golden Case Studies

`case_studies/golden.yaml` is the starting benchmark catalog for the autointerp
agent. Each entry is a public-safe test case with:

- a user-facing research question;
- expected-answer metadata;
- recommended skills;
- suggested methods;
- required evidence;
- known confounds;
- optional public references.

The catalog is intentionally not just prose. It validates through
`autointerp.case_studies` and can later drive automated runs, scoring rubrics,
or benchmark dashboards.

## Private Ground Truth

Some case studies are useful precisely because the target answer is unpublished
or not broadly known. Do not commit those answers to the public repository.

For these cases, the public catalog uses:

```yaml
visibility: private_redacted
expected_answer:
  status: private_redacted
  summary: Private expected mechanism withheld from the public repository.
  redaction_reason: Unpublished result shared for internal supervision only.
```

Store exact private answers in a local overlay under `case_studies/private/`.
That directory is ignored by git.

Suggested local overlay shape:

```yaml
version: "0.1-private"
overrides:
  subliminal-learning-mechanism:
    expected_answer:
      status: internal
      summary: ...
      acceptance_criteria:
        - ...
```

## Current Cases

1. `agentic-misalignment-emotions`: emotion signals before blackmail-like
   behavior.
2. `subliminal-learning-mechanism`: private expected mechanism, public question.
3. `thought-anchors-cot`: load-bearing tokens in chain-of-thought.
4. `surprise-vector`: contrastive surprise representation and steering.
5. `training-free-activation-verbalization`: private expected result, public
   harness task.
6. `ioi-circuit`: canonical indirect-object-identification circuit.
7. `attention-sinks`: BOS or first-token attention sink heads.
8. `binding-ids`: entity-attribute binding codes.
9. `shutdown-resistance-ambiguity`: instruction ambiguity in shutdown scenarios.
10. `assistant-axis`: assistant-role representation and steering.

## Validation

Run:

```bash
python scripts/validate_case_studies.py
```

The validator checks schema validity, the expected number of cases, basic
evaluation scaffolding, and obvious private-result phrases that should not be
committed publicly.

