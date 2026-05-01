#!/usr/bin/env python3
"""Validate skill folder frontmatter and basic hygiene."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "skills"


def validate_skill(path: Path) -> list[str]:
    errors: list[str] = []
    skill_md = path / "SKILL.md"
    if not skill_md.exists():
        return [f"{path.name}: missing SKILL.md"]
    text = skill_md.read_text()
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return [f"{path.name}: invalid frontmatter"]
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return [f"{path.name}: invalid YAML: {exc}"]
    if not isinstance(data, dict):
        return [f"{path.name}: frontmatter must be a mapping"]
    if data.get("name") != path.name:
        errors.append(f"{path.name}: frontmatter name must match folder")
    description = data.get("description")
    if not isinstance(description, str) or len(description.strip()) < 40:
        errors.append(f"{path.name}: description is missing or too short")
    if "TODO" in text:
        errors.append(f"{path.name}: contains TODO placeholder")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(SKILLS_DIR.iterdir()):
        if path.is_dir():
            errors.extend(validate_skill(path))
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Validated {len([p for p in SKILLS_DIR.iterdir() if p.is_dir()])} skills")
    return 0


if __name__ == "__main__":
    sys.exit(main())
