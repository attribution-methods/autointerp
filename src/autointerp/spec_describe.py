"""Render a compact, agent-readable description of the InvestigationSpec schema.

The output is generated from the pydantic models themselves so it cannot drift
from the schema. The agent calls `describe_spec` once at the start of Stage 0
to learn (a) what fields exist, (b) which are required, (c) what types and enum
values are accepted, and (d) the shape of nested types like StageSpec and
Criterion. This replaces the dense paragraphs of allowed-value lists that used
to live in the system prompt.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from autointerp.spec import (
    Approval,
    Budget,
    ContrastSpec,
    CritiqueNote,
    Criterion,
    DatasetSpec,
    InvestigationSpec,
    StageSpec,
)

NESTED_MODELS: list[type[BaseModel]] = [
    StageSpec,
    Criterion,
    DatasetSpec,
    ContrastSpec,
    Budget,
    Approval,
    CritiqueNote,
]


def describe_spec_markdown() -> str:
    sections: list[str] = []
    sections.append("# InvestigationSpec — schema reference\n")
    sections.append(
        "Field names below are the only valid top-level keys for `update_spec`. "
        "Any other key is rejected at patch time. Enum values are listed at "
        "the bottom under \"Closed vocabularies\".\n"
    )
    sections.append("## Top-level fields\n")
    sections.append(_render_model_fields(InvestigationSpec))

    sections.append("\n## Nested types\n")
    for model in NESTED_MODELS:
        sections.append(f"\n### `{model.__name__}`")
        sections.append(_render_model_fields(model))

    sections.append("\n## Closed vocabularies (enum values)\n")
    for enum_cls in _all_enums():
        members = ", ".join(f"`{m.value}`" for m in enum_cls)
        sections.append(f"- **`{enum_cls.__name__}`**: {members}")
    return "\n".join(sections)


def _render_model_fields(model: type[BaseModel]) -> str:
    lines: list[str] = []
    for name, info in model.model_fields.items():
        req = "required" if info.is_required() else "optional"
        type_str = _type_str(info.annotation)
        desc = _field_doc(info)
        suffix = f" — {desc}" if desc else ""
        lines.append(f"- **`{name}`** ({req}, {type_str}){suffix}")
    return "\n".join(lines)


_PIPE = " | "


def _type_str(annotation: Any) -> str:
    if annotation is None or annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            if issubclass(annotation, Enum):
                # Refer by class name so long enums don't bloat the field listing.
                return f"`{annotation.__name__}`"
            if issubclass(annotation, BaseModel):
                return f"`{annotation.__name__}`"
            return annotation.__name__
        return str(annotation)
    args = get_args(annotation)
    if origin is list:
        return f"list[{_type_str(args[0])}]"
    if origin is dict:
        return f"dict[{_type_str(args[0])}, {_type_str(args[1])}]"
    parts = [_type_str(a) for a in args if a is not type(None)]
    if len(args) > len(parts):
        parts = [*parts, "None"]
    # Literal[...] arrives as origin=Literal; render its values inline.
    if str(origin).endswith("Literal"):
        return " | ".join(f"`{a}`" for a in args)
    return " | ".join(parts)


def _field_doc(info: FieldInfo) -> str:
    desc = info.description or ""
    return desc.replace("\n", " ")


def _all_enums() -> list[type[Enum]]:
    seen: list[type[Enum]] = []
    for model in [InvestigationSpec, *NESTED_MODELS]:
        for info in model.model_fields.values():
            for cls in _enums_in(info.annotation):
                if cls not in seen:
                    seen.append(cls)
    return seen


def _enums_in(annotation: Any) -> list[type[Enum]]:
    found: list[type[Enum]] = []
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        found.append(annotation)
        return found
    for arg in get_args(annotation):
        found.extend(_enums_in(arg))
    return found
