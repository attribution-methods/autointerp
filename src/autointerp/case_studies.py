"""Case-study catalog models and loading helpers."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from autointerp.schemas import StrictBaseModel


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE_REDACTED = "private_redacted"
    INTERNAL = "internal"


class ExpectedAnswer(StrictBaseModel):
    """Expected answer metadata for benchmark-style case studies."""

    status: Literal["public", "private_redacted", "hypothesis", "unknown"]
    summary: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    redaction_reason: str | None = None

    @model_validator(mode="after")
    def _private_answers_are_redacted(self) -> "ExpectedAnswer":
        if self.status == "private_redacted" and not self.redaction_reason:
            raise ValueError("private_redacted expected answers require redaction_reason")
        return self


class Reference(StrictBaseModel):
    title: str
    url: str | None = None
    note: str | None = None


class CaseStudySpec(StrictBaseModel):
    case_id: str
    title: str
    question: str
    visibility: Visibility = Visibility.PUBLIC
    difficulty: Literal["smoke", "easy", "medium", "hard", "research"] = "medium"
    evaluation_mode: Literal[
        "black_box",
        "white_box",
        "hybrid",
        "literature_replication",
        "research_open",
    ]
    expected_answer: ExpectedAnswer
    recommended_skills: list[str] = Field(default_factory=list)
    suggested_methods: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    confounds: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    public_notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _visibility_matches_expected_answer(self) -> "CaseStudySpec":
        public_with_private_answer = (
            self.visibility == Visibility.PUBLIC
            and self.expected_answer.status == "private_redacted"
        )
        if public_with_private_answer:
            raise ValueError("public cases cannot have private_redacted expected answers")
        if self.visibility == Visibility.PRIVATE_REDACTED:
            if self.expected_answer.status != "private_redacted":
                raise ValueError(
                    "private_redacted cases must use private_redacted expected answers"
                )
        return self


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

    def public_cases(self) -> list[CaseStudySpec]:
        return [case for case in self.cases if case.visibility == Visibility.PUBLIC]

    def redacted_cases(self) -> list[CaseStudySpec]:
        return [case for case in self.cases if case.visibility == Visibility.PRIVATE_REDACTED]


def load_case_study_catalog(path: str | Path) -> CaseStudyCatalog:
    data = yaml.safe_load(Path(path).read_text())
    return CaseStudyCatalog.model_validate(data)
