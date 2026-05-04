"""In-progress investigation spec used during conversational Stage 0.

The agent builds a `PartialSpec` across turns. When all required fields are
filled and validation passes, it can be finalized into an `InvestigationSpec`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from autointerp.spec import (
    Approval,
    InvestigationSpec,
    SpecStatus,
)

REQUIRED_TOP_LEVEL = (
    "spec_id",
    "question",
    "hypothesis",
    "phenomenon_id",
    "behavior",
    "model",
    "dataset",
    "stages",
    "success_criteria",
)

ALLOWED_TOP_LEVEL = frozenset(InvestigationSpec.model_fields.keys())


class UnknownSpecField(ValueError):
    """Raised when `patch` is called with a key that isn't in InvestigationSpec.

    Carries the offending keys plus a list of valid ones so the agent can
    self-correct without polluting the draft.
    """

    def __init__(self, unknown: list[str]) -> None:
        self.unknown = unknown
        suggestions = []
        for k in unknown:
            close = _suggest_close(k, ALLOWED_TOP_LEVEL)
            if close:
                suggestions.append(f"{k!r} (did you mean {close!r}?)")
            else:
                suggestions.append(repr(k))
        super().__init__(
            "patch contains keys that are not InvestigationSpec fields: "
            + ", ".join(suggestions)
            + ". Allowed top-level fields: "
            + ", ".join(sorted(ALLOWED_TOP_LEVEL))
        )


@dataclass
class PartialSpec:
    """Mutable spec-in-progress. Convert to `InvestigationSpec` on finalize."""

    data: dict[str, Any] = field(default_factory=dict)

    def patch(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply a shallow merge and return a diff of changed top-level fields.

        Raises `UnknownSpecField` if any key isn't a top-level field of
        `InvestigationSpec`. This prevents the agent from polluting the draft
        with typo'd or hallucinated fields (e.g. `title`, `phenomenon`,
        top-level `pattern`) that would only surface as schema errors later.
        """
        unknown = [k for k in updates.keys() if k not in ALLOWED_TOP_LEVEL]
        if unknown:
            raise UnknownSpecField(unknown)
        diff: dict[str, Any] = {}
        for key, new_value in updates.items():
            old_value = self.data.get(key)
            if old_value != new_value:
                diff[key] = {"old": _summarize(old_value), "new": _summarize(new_value)}
                self.data[key] = copy.deepcopy(new_value)
        return diff

    def remove(self, keys: list[str]) -> dict[str, Any]:
        """Delete top-level keys from the draft. Returns what was removed."""
        removed: dict[str, Any] = {}
        for key in keys:
            if key in self.data:
                removed[key] = _summarize(self.data.pop(key), long=True)
        return removed

    def merge_prior(self, prior: dict[str, Any]) -> dict[str, Any]:
        """Merge a phenomenon prior, but do not overwrite fields already set."""
        diff: dict[str, Any] = {}
        for key, value in prior.items():
            if key not in self.data:
                self.data[key] = copy.deepcopy(value)
                diff[key] = {"old": None, "new": _summarize(value)}
        return diff

    def missing_required(self) -> list[str]:
        return [k for k in REQUIRED_TOP_LEVEL if k not in self.data]

    def as_markdown(self) -> str:
        """Render the FULL InvestigationSpec shape, with `_(unset)_` for fields
        not yet filled. Every render doubles as a schema reminder so the agent
        always sees the target shape it's working towards.
        """
        lines = ["# Investigation Spec (draft)\n"]
        required_set = set(REQUIRED_TOP_LEVEL)
        # Required first, in canonical order, then optional fields, then any
        # stray keys (which shouldn't exist post-typed-gate but render anyway).
        for key in REQUIRED_TOP_LEVEL:
            value = self.data.get(key)
            marker = "" if key in self.data else "  *(required, unset)*"
            shown = _summarize(value, long=True) if key in self.data else "_(unset)_"
            lines.append(f"**{key}**: {shown}{marker}")
        optional = [k for k in ALLOWED_TOP_LEVEL if k not in required_set]
        for key in sorted(optional):
            if key in self.data:
                lines.append(f"**{key}**: {_summarize(self.data[key], long=True)}")
            else:
                lines.append(f"**{key}**: _(unset, optional)_")
        stray = [k for k in self.data.keys() if k not in ALLOWED_TOP_LEVEL]
        for key in stray:
            lines.append(
                f"**{key}** *(not in schema!)*: {_summarize(self.data[key], long=True)}"
            )
        return "\n".join(lines)

    def try_build(self, status: SpecStatus = SpecStatus.DRAFT) -> InvestigationSpec:
        """Build an `InvestigationSpec` from the current state.

        Raises `pydantic.ValidationError` if the partial spec is incomplete or
        any field fails its validator. Caller is expected to catch and report.
        """
        payload = dict(self.data)
        payload.setdefault("status", status.value)
        return InvestigationSpec.model_validate(payload)

    def reset(self) -> None:
        self.data.clear()


def _suggest_close(needle: str, candidates) -> str | None:
    """Return the closest candidate key by simple substring/prefix match."""
    needle_l = needle.lower()
    matches = [
        c for c in candidates
        if needle_l in c.lower() or c.lower().startswith(needle_l[:4])
    ]
    matches.sort(key=lambda c: abs(len(c) - len(needle)))
    return matches[0] if matches else None


def _summarize(value: Any, long: bool = False) -> str:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        return text if long or len(text) <= 80 else text[:77] + "..."
    if isinstance(value, list):
        return f"[{len(value)} item(s)]" if not long else str(value)[:400]
    if isinstance(value, dict):
        keys = ", ".join(list(value.keys())[:6])
        return f"{{{keys}}}" if not long else str(value)[:400]
    return type(value).__name__


def make_approval(approver: str, kind: str, notes: str | None = None) -> Approval:
    return Approval(
        approver=approver,
        approver_kind=kind,  # type: ignore[arg-type]
        approved_at=datetime.now(timezone.utc),
        notes=notes,
    )
