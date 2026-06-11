"""Tests for the per-model cost accumulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autointerp.utils.cost import CostTracker


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _Response:
    model: str
    usage: _Usage
    _hidden_params: dict[str, Any]


def _resp(model: str, prompt: int, completion: int, cost: float | None = 0.0) -> _Response:
    return _Response(
        model=model,
        usage=_Usage(prompt, completion),
        _hidden_params={"response_cost": cost},
    )


def test_accumulates_per_model() -> None:
    t = CostTracker(run_id="r1")
    t.add_response(_resp("anthropic/claude-sonnet-4-6", 100, 50, 0.01))
    t.add_response(_resp("anthropic/claude-sonnet-4-6", 200, 75, 0.02))
    t.add_response(_resp("anthropic/claude-haiku-4-5", 10, 5, 0.001))

    sonnet = t.by_model["anthropic/claude-sonnet-4-6"]
    assert sonnet.input_tokens == 300
    assert sonnet.output_tokens == 125
    assert sonnet.requests == 2
    assert abs(sonnet.cost_usd - 0.03) < 1e-9
    assert t.total_requests == 3
    assert abs(t.total_cost_usd - 0.031) < 1e-9


def test_unknown_cost_flag() -> None:
    t = CostTracker(run_id="r1")
    t.add_response(_resp("local/whatever", 100, 50, cost=None))
    assert t.has_unknown_cost is True
    assert t.total_cost_usd == 0.0


def test_atomic_write_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    t = CostTracker(run_id="run-A")
    t.add_response(_resp("anthropic/claude-sonnet-4-6", 100, 50, 0.05))
    t.write(path)

    raw = json.loads(path.read_text())
    assert raw["run_id"] == "run-A"
    assert raw["total_cost_usd"] == 0.05

    # Resume: same run_id reads back accumulators.
    t2 = CostTracker.load_or_new(path, run_id="run-A")
    assert t2.total_requests == 1
    assert t2.total_cost_usd == 0.05
    assert t2.by_model["anthropic/claude-sonnet-4-6"].input_tokens == 100

    # Mismatched run_id resets — same convention as cost-tracker.ts.
    t3 = CostTracker.load_or_new(path, run_id="run-B")
    assert t3.total_requests == 0


def test_format_summary() -> None:
    t = CostTracker(run_id="r")
    t.add_response(_resp("anthropic/claude-sonnet-4-6", 1000, 500, 0.0123))
    out = t.format_summary()
    assert "Total cost: $0.0123" in out
    assert "anthropic/claude-sonnet-4-6" in out
    assert "1,000" in out


def test_handles_dict_response() -> None:
    """LiteLLM sometimes hands back a dict-like response — accept it."""
    t = CostTracker(run_id="r")
    t.add_response(
        {
            "model": "x/y",
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            "_hidden_params": {"response_cost": 0.01},
        }
    )
    assert t.by_model["x/y"].input_tokens == 5
    assert t.total_cost_usd == 0.01


def test_safe_against_missing_usage() -> None:
    t = CostTracker(run_id="r")
    # Response with no usage field — must not crash.
    t.add_response({"model": "x", "usage": None, "_hidden_params": {}})
    assert t.total_requests == 0
