"""Task / config / result types for the generic hill-climbing engine.

The engine (``engine.run_hillclimb``) optimizes any artifact against any
reward, given:

- a ``HillClimbTask`` — *what* is being optimized (the artifact contract, a
  seed template, the reward metric, and instructions for the proposer);
- a ``HillClimbConfig`` — *how hard* to search (iterations, width, archive,
  propose mode, budget).

``discover_features`` is one instantiation (see ``discovery.py``). Adding a new
optimization target means building a different ``HillClimbTask`` — no change to
the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HillClimbTask:
    """What is being optimized."""

    name: str
    task_text: str
    reward_metric: str
    reward_description: str
    # Seed artifact: copied to candidate_v1 and shown as the starting point.
    candidate_template: Path
    # Contract the proposer must honor (appended to the system prompt) — e.g.
    # "define `score(...)` returning list[Candidate] sorted by abs(score)".
    contract_instructions: str
    # File extension of a candidate artifact (".py" for code, ".txt"/".md" for
    # prompts, ...). The proposer writes candidate_v{N}{artifact_suffix}.
    artifact_suffix: str = ".py"


@dataclass
class HillClimbConfig:
    """How hard to search. Mirrors the spec's DiscoveryConfig at runtime."""

    max_iterations: int = 8
    patience: int = 2
    # Parallel proposals per round (width). 1 = pure single-branch hill climb.
    n_subagents: int = 1
    # Top-K candidates kept as the archive (population). >=1.
    archive_size: int = 5
    # How many top candidates' FULL code to inline in the proposal prompt
    # (besides the parent). The rest appear only as a compact leaderboard;
    # agentic proposers read full code from disk on demand. 0 = parent only.
    archive_code_in_context: int = 1
    # "single" = one LLM completion per proposal; "agentic" = a tool-using
    # subagent via run_agent_turn.
    propose_mode: str = "single"
    max_turns_per_iteration: int = 20  # agentic only
    model: str = "anthropic/claude-sonnet-4-5"
    temperature: float | None = None
    # Variance seeds forwarded to the evaluator (re-evaluate top candidate
    # across seeds to screen out flukes). Empty/[0] => single evaluation.
    seeds: list[int] = field(default_factory=lambda: [0])
    # Evaluator knobs.
    top_k: int = 20
    k_grid: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    objective: str = "combined"
    evaluator: str | None = None
    # Budget cap (output+input tokens) for this whole discovery call. None =
    # uncapped. The engine stops proposing once exceeded.
    max_tokens: int | None = None
    dry_run: bool = False


@dataclass
class HillClimbResult:
    """Final hand-back to the caller."""

    session_dir: Path
    best_candidate: str | None
    best_reward: float | None
    best_summary: dict[str, Any] | None
    # Top-K archive: [{"candidate", "reward", "summary"}], best first.
    archive: list[dict[str, Any]]
    iterations_run: int
    proposals_made: int
    terminated_by: str  # converged | max_iters | perfect | budget | error
    tokens_used: int = 0
    cost_usd: float = 0.0
    # Variance of the best candidate's reward across seeds (None if 1 seed).
    best_reward_std: float | None = None
    log: list[dict[str, Any]] = field(default_factory=list)


__all__ = ["HillClimbTask", "HillClimbConfig", "HillClimbResult"]
