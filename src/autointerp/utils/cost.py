"""Per-model usage and cost accumulator.

Adapted from the per-model accumulator in claude-code's ``cost-tracker.ts``:
one ``ModelUsage`` record per ``model_name`` carrying tokens (input/output,
cache read/cache creation) and USD. Persisted to ``cost.json`` in the run
directory; restored on resume so multi-process resumes carry forward.

LiteLLM exposes ``response.usage`` and ``response._hidden_params['response_cost']``
on most providers — we read both, fall back to ``litellm.completion_cost``
when the cost field is missing, and tolerate "unknown model" gracefully.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    """Per-run cumulative cost. Atomic writes; thread-safe accumulators."""

    run_id: str
    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_requests: int = 0
    has_unknown_cost: bool = False
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @property
    def total_tokens(self) -> int:
        """Input + output tokens across all models (cache tokens excluded)."""
        return sum(
            mu.input_tokens + mu.output_tokens for mu in self.by_model.values()
        )

    def add_response(self, response: Any) -> None:
        """Accumulate from a LiteLLM response. Silent on missing fields."""
        model = _safe_attr(response, "model") or "unknown"
        usage = _safe_attr(response, "usage")
        if usage is None:
            return
        in_tok = int(_safe_attr(usage, "prompt_tokens") or 0)
        out_tok = int(_safe_attr(usage, "completion_tokens") or 0)
        cache_read = 0
        cache_create = 0
        # Anthropic-style nested usage details (when LiteLLM passes them through).
        details = _safe_attr(usage, "prompt_tokens_details")
        if details is not None:
            cache_read = int(_safe_attr(details, "cached_tokens") or 0)
        cache_create = int(_safe_attr(usage, "cache_creation_input_tokens") or 0)
        cache_read = max(
            cache_read, int(_safe_attr(usage, "cache_read_input_tokens") or 0)
        )

        cost = _extract_cost(response)
        with self._lock:
            mu = self.by_model.setdefault(model, ModelUsage())
            mu.input_tokens += in_tok
            mu.output_tokens += out_tok
            mu.cache_read_input_tokens += cache_read
            mu.cache_creation_input_tokens += cache_create
            mu.requests += 1
            if cost is None:
                self.has_unknown_cost = True
            else:
                mu.cost_usd += cost
                self.total_cost_usd += cost
            self.total_requests += 1

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_requests": self.total_requests,
            "has_unknown_cost": self.has_unknown_cost,
            "by_model": {
                name: {**asdict(usage), "cost_usd": round(usage.cost_usd, 6)}
                for name, usage in self.by_model.items()
            },
        }

    def write(self, path: Path) -> None:
        """Atomic write to ``path`` (tempfile + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        tmp.replace(path)

    @classmethod
    def load_or_new(cls, path: Path, run_id: str) -> "CostTracker":
        """Resume from ``path`` if present, else return a fresh tracker.

        Mismatched run_id resets the tracker — same convention as
        cost-tracker.ts ``getStoredSessionCosts``: only restore if identity matches.
        """
        path = Path(path)
        if not path.exists():
            return cls(run_id=run_id)
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls(run_id=run_id)
        if data.get("run_id") != run_id:
            return cls(run_id=run_id)
        tracker = cls(
            run_id=run_id,
            total_cost_usd=float(data.get("total_cost_usd", 0.0)),
            total_requests=int(data.get("total_requests", 0)),
            has_unknown_cost=bool(data.get("has_unknown_cost", False)),
        )
        for name, raw in (data.get("by_model") or {}).items():
            tracker.by_model[name] = ModelUsage(
                input_tokens=int(raw.get("input_tokens", 0)),
                output_tokens=int(raw.get("output_tokens", 0)),
                cache_read_input_tokens=int(raw.get("cache_read_input_tokens", 0)),
                cache_creation_input_tokens=int(
                    raw.get("cache_creation_input_tokens", 0)
                ),
                requests=int(raw.get("requests", 0)),
                cost_usd=float(raw.get("cost_usd", 0.0)),
            )
        return tracker

    # -- formatting --------------------------------------------------------

    def format_summary(self) -> str:
        if self.total_requests == 0:
            return "Cost: 0 requests"
        lines = [
            f"Total cost: ${self.total_cost_usd:.4f}"
            + ("  (some models missing pricing)" if self.has_unknown_cost else ""),
            f"Requests:   {self.total_requests}",
        ]
        for name, mu in sorted(self.by_model.items()):
            lines.append(
                f"  {name}: {mu.requests} req, "
                f"{mu.input_tokens:,} in / {mu.output_tokens:,} out, "
                f"{mu.cache_read_input_tokens:,} cache read / "
                f"{mu.cache_creation_input_tokens:,} cache write"
                f"  (${mu.cost_usd:.4f})"
            )
        return "\n".join(lines)


def _safe_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_cost(response: Any) -> float | None:
    # LiteLLM stamps ``_hidden_params['response_cost']`` on the response when
    # pricing is known. Fall back to ``litellm.completion_cost``; on unknown
    # models / providers it raises — return None so the tracker flags it.
    hidden = _safe_attr(response, "_hidden_params")
    if hidden is not None:
        cost = _safe_attr(hidden, "response_cost")
        if cost is not None:
            try:
                return float(cost)
            except (TypeError, ValueError):
                pass
    try:
        from litellm import completion_cost
    except Exception:
        return None
    try:
        return float(completion_cost(completion_response=response))
    except Exception:
        return None


__all__ = ["CostTracker", "ModelUsage"]
