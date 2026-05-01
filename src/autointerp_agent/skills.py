"""Skill discovery and loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    body: str


class SkillRegistry:
    def __init__(self, skills: dict[str, Skill]):
        self.skills = skills

    @classmethod
    def from_dir(cls, skills_dir: str | Path) -> "SkillRegistry":
        root = Path(skills_dir)
        skills: dict[str, Skill] = {}
        if not root.exists():
            return cls(skills)
        for skill_md in sorted(root.glob("*/SKILL.md")):
            skill = _read_skill(skill_md)
            skills[skill.name] = skill
        return cls(skills)

    @classmethod
    def from_repo(cls) -> "SkillRegistry":
        env_root = os.environ.get("AUTOINTERP_SKILLS_DIR")
        if env_root:
            return cls.from_dir(env_root)
        root = Path(__file__).resolve().parents[2] / "skills"
        if not root.exists():
            root = Path.cwd() / "skills"
        return cls.from_dir(root)

    def get(self, name: str) -> Skill:
        return self.skills[name]

    def list_lines(self) -> list[str]:
        return [f"- {skill.name}: {skill.description}" for skill in self.skills.values()]

    def select(self, names: Iterable[str]) -> list[Skill]:
        return [self.skills[name] for name in names if name in self.skills]


def _read_skill(path: Path) -> Skill:
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid skill frontmatter: {path}")
    frontmatter = yaml.safe_load(match.group(1))
    if not isinstance(frontmatter, dict):
        raise ValueError(f"Invalid skill frontmatter mapping: {path}")
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError(f"Skill missing name or description: {path}")
    return Skill(name=name, description=description, path=path, body=match.group(2).strip())
