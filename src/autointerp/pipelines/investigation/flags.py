"""Ablation feature-flags for the scaffold-faithfulness evaluation.

The investigation pipeline normally enforces every discipline mechanism. For
the meta-evaluation (see ``docs/scaffold_faithfulness_eval.md``) we need to
disable exactly one mechanism at a time while holding everything else
constant. Three flags, each mapping to one mechanism and one error mode:

- ``freeze_spec`` (A): spec.json is chmod read-only and the spec is declared
  immutable. Off → the spec is writable; the agent can redefine success
  after seeing data (HARKing / post-hoc rationalization).
- ``provenance_metrics`` (B): ``MetricResult`` requires a one-time provenance
  token from ``compute_metric``; the value cannot be edited. Off → the agent
  may self-report metric values (number fabrication).
- ``split_disjoint`` (C): a criterion with ``on_split=heldout`` refuses
  MetricResults tagged with another split. Off → discovery/validation
  contamination (circular validation).

**Default is all-on**, which is byte-for-byte the current scaffold. Disabling
a flag removes both the mechanical gate *and* its announced rule in the
system prompt (decision D2): the run genuinely lacks that discipline rather
than being told a rule that is no longer enforced.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# (flag attribute, short ablation code) — order is the report/label order.
_FLAG_CODES: tuple[tuple[str, str], ...] = (
    ("freeze_spec", "A"),
    ("provenance_metrics", "B"),
    ("split_disjoint", "C"),
)


class AblationFlags(BaseModel):
    """Immutable set of discipline toggles for one investigation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    freeze_spec: bool = True
    provenance_metrics: bool = True
    split_disjoint: bool = True

    @property
    def all_on(self) -> bool:
        return self.freeze_spec and self.provenance_metrics and self.split_disjoint

    def label(self) -> str:
        """Stable condition label, e.g. ``full`` or ``loo-B``."""
        if self.all_on:
            return "full"
        off = [code for attr, code in _FLAG_CODES if not getattr(self, attr)]
        return "loo-" + "".join(off)

    @classmethod
    def leave_out(cls, code: str) -> "AblationFlags":
        """Build a leave-one-out config: every flag on except ``code`` (A/B/C)."""
        code = code.strip().upper()
        attr = {c: a for a, c in _FLAG_CODES}.get(code)
        if attr is None:
            raise ValueError(f"unknown ablation code {code!r}; valid: A, B, C")
        return cls(**{attr: False})

    @classmethod
    def parse(cls, spec: str | None) -> "AblationFlags":
        """Parse a CLI spec: ``full`` / ``none`` / ``A`` / ``A,C`` (= off codes)."""
        if spec is None:
            return cls()
        s = spec.strip().lower()
        if s in ("", "full", "all", "all-on"):
            return cls()
        off = {tok.strip().upper() for tok in spec.split(",") if tok.strip()}
        unknown = off - {c for _, c in _FLAG_CODES}
        if unknown:
            raise ValueError(f"unknown ablation code(s) {sorted(unknown)}; valid: A, B, C")
        return cls(**{attr: code not in off for attr, code in _FLAG_CODES})


__all__ = ["AblationFlags"]
