#!/usr/bin/env python3
"""Validate public case-study catalogs."""

from __future__ import annotations

import sys
from pathlib import Path

from autointerp.case_studies import Visibility, load_case_study_catalog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "case_studies" / "golden.yaml"
PUBLIC_REPO_FORBIDDEN = [
    "single linear direction",
    "single direction",
    "pathscopes works surprisingly well",
    "much better than what is reported",
]


def validate_catalog(path: Path) -> list[str]:
    errors: list[str] = []
    catalog = load_case_study_catalog(path)
    if len(catalog.cases) != 10:
        errors.append(f"{path}: expected 10 golden cases, found {len(catalog.cases)}")
    for case in catalog.cases:
        if not case.recommended_skills:
            errors.append(f"{case.case_id}: missing recommended_skills")
        if not case.required_evidence:
            errors.append(f"{case.case_id}: missing required_evidence")
        if case.visibility == Visibility.PRIVATE_REDACTED:
            if case.expected_answer.status != "private_redacted":
                errors.append(f"{case.case_id}: private case has non-private expected answer")
            if not case.expected_answer.redaction_reason:
                errors.append(f"{case.case_id}: private case missing redaction reason")
    text = path.read_text().lower()
    for phrase in PUBLIC_REPO_FORBIDDEN:
        if phrase in text:
            errors.append(f"{path}: contains private or over-specific phrase: {phrase!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (argv or sys.argv[1:])] or [DEFAULT_CATALOG]
    errors: list[str] = []
    for path in paths:
        errors.extend(validate_catalog(path))
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

