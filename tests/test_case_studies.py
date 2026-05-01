from autointerp.case_studies import Visibility, load_case_study_catalog


def test_golden_catalog_loads():
    catalog = load_case_study_catalog("case_studies/golden.yaml")

    assert len(catalog.cases) == 10
    assert catalog.cases[0].case_id == "agentic-misalignment-emotions"
    assert len(catalog.redacted_cases()) == 2
    assert len(catalog.public_cases()) == 8


def test_private_expected_answers_are_redacted():
    catalog = load_case_study_catalog("case_studies/golden.yaml")
    private_cases = {case.case_id: case for case in catalog.redacted_cases()}

    assert "subliminal-learning-mechanism" in private_cases
    assert "training-free-activation-verbalization" in private_cases
    for case in private_cases.values():
        assert case.visibility == Visibility.PRIVATE_REDACTED
        assert case.expected_answer.status == "private_redacted"
        assert case.expected_answer.redaction_reason
        assert "withheld" in case.expected_answer.summary.lower()


def test_each_case_has_eval_scaffolding():
    catalog = load_case_study_catalog("case_studies/golden.yaml")

    for case in catalog.cases:
        assert case.question.endswith("?")
        assert case.recommended_skills
        assert case.suggested_methods
        assert case.required_evidence
        assert case.expected_answer.acceptance_criteria

