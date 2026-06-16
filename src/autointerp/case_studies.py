"""Case-study catalog models and loaders.

The catalog has two deliberately separated halves:

* The PUBLIC catalog (``case_studies/golden.yaml``) is spoiler-free. It holds the
  research QUESTION and methodology for each case — never the expected answer.
  ``CaseStudySpec`` has no answer field at all, and because it is a
  ``StrictBaseModel`` (extra keys forbidden), an answer accidentally added to the
  public YAML fails validation. This is what makes the catalog safe to keep in
  the repo: the investigation agent can read any file in its checkout, so the
  ground truth must simply not be there.

* The PRIVATE ground truth (``GroundTruthSet``) holds the expected answers. It
  lives in a SEPARATE private store, loaded on demand from a path outside the
  public catalog (env var or a git-ignored default) and is never committed. For a
  real evaluation it must be absent on the machine where the agent runs, so the
  agent cannot reach it and reward-hack. See ``docs/case_studies.md``.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from autointerp.schemas import StrictBaseModel


class AnswerStatus(str, Enum):
    """Whether a case's ground truth is publicly known or an internal result.

    This is metadata about the ANSWER, kept on the public case so a showcase can
    treat the two differently — it never reveals the answer itself.
    """

    PUBLISHED = "published"      # established in the literature (e.g. IOI, attention sinks)
    UNPUBLISHED = "unpublished"  # internal/unpublished result — keep out of any public artifact


EvaluationMode = Literal[
    "black_box",
    "white_box",
    "hybrid",
    "literature_replication",
    "research_open",
]
Difficulty = Literal["smoke", "easy", "medium", "hard", "research"]


class CaseStudySpec(StrictBaseModel):
    """One public-safe case study: a research question plus methodology.

    Holds NO expected answer — that lives only in the private ground-truth store.
    """

    case_id: str
    title: str
    question: str
    difficulty: Difficulty = "medium"
    evaluation_mode: EvaluationMode
    answer_status: AnswerStatus = AnswerStatus.PUBLISHED
    recommended_skills: list[str] = Field(default_factory=list)
    suggested_methods: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    confounds: list[str] = Field(default_factory=list)
    public_notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseStudyCatalog(StrictBaseModel):
    version: str
    description: str
    cases: list[CaseStudySpec]

    @model_validator(mode="after")
    def _unique_case_ids(self) -> "CaseStudyCatalog":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique")
        return self

    def case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases]

    def get(self, case_id: str) -> CaseStudySpec | None:
        return next((c for c in self.cases if c.case_id == case_id), None)

    def published(self) -> list[CaseStudySpec]:
        return [c for c in self.cases if c.answer_status == AnswerStatus.PUBLISHED]

    def unpublished(self) -> list[CaseStudySpec]:
        return [c for c in self.cases if c.answer_status == AnswerStatus.UNPUBLISHED]


def load_case_study_catalog(path: str | Path) -> CaseStudyCatalog:
    data = yaml.safe_load(Path(path).read_text())
    return CaseStudyCatalog.model_validate(data)


# ---------------------------------------------------------------------------
# Private ground truth — loaded from OUTSIDE the public catalog, never committed.
# ---------------------------------------------------------------------------

#: Environment variable pointing at the private ground-truth YAML. Set this to a
#: path in your separate private store. If unset, ``ground_truth_path`` falls back
#: to the git-ignored default below (handy locally; absent in a clean checkout).
GROUND_TRUTH_ENV = "AUTOINTERP_CASE_STUDY_GROUND_TRUTH"
_DEFAULT_PRIVATE_PATH = Path("case_studies/private/ground_truth.yaml")


class GroundTruth(StrictBaseModel):
    """Expected answer for ONE case. Private — never committed, never on a machine
    where the agent runs."""

    case_id: str
    summary: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    source: AnswerStatus = AnswerStatus.PUBLISHED
    notes: str | None = None


class GroundTruthSet(StrictBaseModel):
    version: str
    answers: dict[str, GroundTruth]

    @model_validator(mode="after")
    def _keys_match_case_ids(self) -> "GroundTruthSet":
        for key, gt in self.answers.items():
            if gt.case_id != key:
                raise ValueError(f"answers[{key!r}].case_id is {gt.case_id!r}; must match the key")
        return self

    def get(self, case_id: str) -> GroundTruth | None:
        return self.answers.get(case_id)


def ground_truth_path() -> Path | None:
    """Resolve the private ground-truth file: the env override if set, else the
    git-ignored default if it happens to exist locally, else ``None``.

    Returns ``None`` (not an error) when no private store is present — the public
    catalog and its tooling work without it.
    """
    env = os.environ.get(GROUND_TRUTH_ENV)
    if env:
        return Path(env)
    return _DEFAULT_PRIVATE_PATH if _DEFAULT_PRIVATE_PATH.exists() else None


def load_ground_truth(path: str | Path | None = None) -> GroundTruthSet | None:
    """Load the private ground-truth set, or ``None`` if no private store exists.

    Used only by OFFLINE scoring/showcase tooling — never by the agent runtime.
    """
    resolved = Path(path) if path is not None else ground_truth_path()
    if resolved is None or not resolved.exists():
        return None
    return GroundTruthSet.model_validate(yaml.safe_load(resolved.read_text()))
