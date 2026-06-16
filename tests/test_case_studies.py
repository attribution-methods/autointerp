"""Schema tests for the public case-study catalog + the ground-truth template.

The point of these is structural: the public catalog must stay spoiler-free
(no answer field can exist on a case) and well-formed. We never load the real
private ground truth here.
"""

import pytest

from autointerp.case_studies import (
    CaseStudySpec,
    GroundTruthSet,
    load_case_study_catalog,
)

CATALOG = "case_studies/golden.yaml"


def test_golden_catalog_loads() -> None:
    catalog = load_case_study_catalog(CATALOG)
    assert len(catalog.cases) == 10
    assert len(set(catalog.case_ids())) == 10  # unique
    assert catalog.cases[0].case_id == "emotions-before-misalignment"


def test_public_catalog_has_no_answer_field() -> None:
    """The spec carries no expected answer, and forbids extra keys — so an answer
    smuggled into golden.yaml would fail to load. This is the out-of-reach guard."""
    fields = set(CaseStudySpec.model_fields)
    assert not (fields & {"expected_answer", "answer", "ground_truth", "summary"})
    # StrictBaseModel forbids extras: a stray answer key is rejected, not ignored.
    with pytest.raises(Exception):
        CaseStudySpec.model_validate({
            "case_id": "x", "title": "x", "question": "x?",
            "evaluation_mode": "white_box", "expected_answer": "leak",
        })


def test_each_case_has_scaffolding_but_no_spoilers() -> None:
    catalog = load_case_study_catalog(CATALOG)
    for case in catalog.cases:
        assert case.question.endswith("?")
        assert case.recommended_skills
        assert case.suggested_methods
        assert case.required_evidence
        assert case.answer_status in {"published", "unpublished"}


def test_unpublished_cases_are_flagged() -> None:
    catalog = load_case_study_catalog(CATALOG)
    unpublished = {c.case_id for c in catalog.unpublished()}
    # The internal/unpublished results must be marked so showcases can redact them.
    assert {"subliminal-learning-mechanism", "being-the-assistant-representation"} <= unpublished


def test_ground_truth_template_is_valid_and_answerless() -> None:
    """The committed template parses as a GroundTruthSet (shape only) and ships no
    real answer (placeholders use angle brackets)."""
    import yaml
    text = open("case_studies/ground_truth.example.yaml").read()
    GroundTruthSet.model_validate(yaml.safe_load(text))
    assert "<" in text and ">" in text  # placeholders, not real content
