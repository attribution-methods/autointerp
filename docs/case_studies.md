# Golden Case Studies

A catalog of 10 interpretability questions used to evaluate the autointerp agent.
It has two deliberately separated halves.

## Why the split

The investigation agent can read **any file in its checkout** during a run
(`read_file` and `bash` are not sandboxed for reads). So if a ground-truth answer
lived anywhere in the repo, the agent could `cat` it and reward-hack. The only
robust "out of reach" is **"not present."** Therefore:

- **Public, spoiler-free catalog** — `case_studies/golden.yaml`. The research
  **question** and methodology for each case: recommended skills, suggested
  methods, required evidence, confounds, difficulty, tags. **No expected answer.**
  `CaseStudySpec` has no answer field and forbids extra keys, so an answer added
  here fails schema validation. Even `case_id`s and titles are phrased from the
  *question*, never the answer (a case about where attention concentrates is
  `where-attention-concentrates`, not its answer).

- **Private ground truth** — the expected answers, kept in a **separate private
  store**, never committed. Loaded on demand by offline tooling only.

This is not wired into the agent runtime (`src/autointerp_agent/` never imports
`case_studies`), so it cannot influence a run.

## The private ground-truth store

The answers live in a `GroundTruthSet` (see `case_studies/ground_truth.example.yaml`
for the shape — a committed template with placeholders, no real content). For a
real, adversarial evaluation the file **must be absent on the machine where the
agent runs.**

Resolution order (offline tooling, `autointerp.case_studies.load_ground_truth`):

1. `AUTOINTERP_CASE_STUDY_GROUND_TRUTH` env var — point it at your private store;
2. else the git-ignored default `case_studies/private/ground_truth.yaml` if it
   happens to exist locally;
3. else `None` — the catalog and its tooling work fine without it.

To set up locally:

```bash
cp case_studies/ground_truth.example.yaml case_studies/private/ground_truth.yaml
# fill in real answers; case_studies/private/ is git-ignored
```

`answer_status` on each public case (`published` / `unpublished`) flags which
ground truths are internal/unpublished (e.g. subliminal learning, the assistant
representation) so a public showcase can redact them — without revealing them.

## The 10 questions (answers withheld)

1. `emotions-before-misalignment` — what model emotions precede misaligned behavior?
2. `subliminal-learning-mechanism` — how does subliminal learning work? *(unpublished)*
3. `load-bearing-cot-tokens` — are there load-bearing tokens in chain-of-thought?
4. `surprise-representation` — how do models represent surprise in text?
5. `training-free-activation-verbalization` — a training-free way to verbalize
   the concepts in activations? *(unpublished)*
6. `indirect-object-identification` — how do models do IOI?
7. `where-attention-concentrates` — where does generation-time attention concentrate, and why?
8. `entity-attribute-tracking` — how does a model track which entity has which item?
9. `shutdown-resistance` — why do models sometimes resist mid-execution shutdown?
10. `being-the-assistant-representation` — is there a unified "being the assistant"
    representation? *(unpublished)*

## Findings & comparison

`case_studies/findings/` records **what autointerp currently finds** for each
question (the agent's own outputs — safe to commit; they reveal no ground truth).
A public compare-&-contrast showcase summarizes how the agent did at a high level;
the detailed scoring against the private ground truth stays in the private store.

## Validation

```bash
python scripts/validate_case_studies.py
```

Schema only: the catalog is well-formed, has 10 cases, each is a question with
methodology, and the committed ground-truth template parses. It never loads the
real private answers.
