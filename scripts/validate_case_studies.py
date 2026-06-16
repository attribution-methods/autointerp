#!/usr/bin/env python3
"""Validate the public case-study catalog (schema only).

The catalog is spoiler-free by construction: ``CaseStudySpec`` has no answer
field and forbids extra keys, so an expected answer accidentally added to
``golden.yaml`` fails to load here. We also schema-check the committed
ground-truth TEMPLATE so its shape stays valid. We never load or print the real
private ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from autointerp.case_studies import (
    CaseStudyCatalog,
    GroundTruthSet,
    load_case_study_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "case_studies" / "golden.yaml"
GROUND_TRUTH_TEMPLATE = ROOT / "case_studies" / "ground_truth.example.yaml"


def validate_catalog(path: Path) -> list[str]:
    errors: list[str] = []
    catalog: CaseStudyCatalog = load_case_study_catalog(path)
    if len(catalog.cases) != 10:
        errors.append(f"{path}: expected 10 golden cases, found {len(catalog.cases)}")
    for case in catalog.cases:
        if not case.question.strip().endswith("?"):
            errors.append(f"{case.case_id}: question must be phrased as a question")
        for field in ("recommended_skills", "suggested_methods", "required_evidence"):
            if not getattr(case, field):
                errors.append(f"{case.case_id}: missing {field}")
    return errors


def validate_template(path: Path) -> list[str]:
    """Schema-check the committed ground-truth template (shape only)."""
    if not path.exists():
        return [f"{path}: missing ground-truth template"]
    try:
        GroundTruthSet.model_validate(yaml.safe_load(path.read_text()))
    except Exception as exc:  # noqa: BLE001 — surface the validation reason
        return [f"{path}: invalid ground-truth template: {exc}"]
    return []


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (argv or sys.argv[1:])] or [DEFAULT_CATALOG]
    errors: list[str] = []
    for path in paths:
        errors.extend(validate_catalog(path))
    if DEFAULT_CATALOG in paths:
        errors.extend(validate_template(GROUND_TRUTH_TEMPLATE))
    if errors:
        for error in errors:
            print(error)
        return 1
    for path in paths:
        catalog = load_case_study_catalog(path)
        print(f"Validated {len(catalog.cases)} case studies in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
