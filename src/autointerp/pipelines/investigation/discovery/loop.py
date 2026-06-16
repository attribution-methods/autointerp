"""Feature-discovery wrapper over the generic hill-climbing engine.

``discover_features`` is one instantiation of ``engine.run_hillclimb``: the
artifact is a ``score(...)`` ranking function, the reward is an AUC-K metric,
and the evaluator is the discovery harness. Adding another optimization target
(prompt search, probe search, ...) means building a different ``HillClimbTask``
and calling ``run_hillclimb`` directly — no new engine.
"""

from __future__ import annotations

from pathlib import Path

from .engine import run_hillclimb
from .task import HillClimbConfig, HillClimbResult, HillClimbTask

_TEMPLATE = Path(__file__).resolve().parent / "algorithm_template.py"

DISCOVERY_CONTRACT = (
    "Define a top-level `def score(pairs, *, loader, layers, device, top_k, "
    "context=None)` that ranks model components/features and returns a "
    "`list[Candidate]` (from "
    "`autointerp.pipelines.investigation.discovery.candidate`), sorted by "
    "descending `abs(score)`, length <= top_k. Pure ranking only: no I/O, no "
    "global mutation; the harness owns model loading, ablation, steering, and "
    "metric computation."
)

# Back-compat alias: callers/tests may import DiscoveryResult.
DiscoveryResult = HillClimbResult


async def run_discovery_subagent(
    *,
    session_dir: Path,
    task: str,
    reward_metric: str = "combined_auc_k",
    reward_description: str = "",
    model: str = "anthropic/claude-sonnet-4-5",
    temperature: float | None = None,
    max_iterations: int = 8,
    patience: int = 2,
    n_subagents: int = 1,
    archive_size: int = 5,
    archive_code_in_context: int = 1,
    propose_mode: str = "single",
    max_turns_per_iteration: int = 20,
    seeds: list[int] | None = None,
    top_k: int = 20,
    k_grid: list[int] | None = None,
    objective: str = "combined",
    evaluator: str | None = None,
    max_tokens: int | None = None,
    dry_run: bool = False,
) -> HillClimbResult:
    """Run feature discovery as a hill-climbing search (async)."""
    hc_task = HillClimbTask(
        name="feature_discovery",
        task_text=task,
        reward_metric=reward_metric,
        reward_description=reward_description,
        candidate_template=_TEMPLATE,
        contract_instructions=DISCOVERY_CONTRACT,
        artifact_suffix=".py",
    )
    hc_cfg = HillClimbConfig(
        max_iterations=max_iterations,
        patience=patience,
        n_subagents=n_subagents,
        archive_size=archive_size,
        archive_code_in_context=archive_code_in_context,
        propose_mode=propose_mode,
        max_turns_per_iteration=max_turns_per_iteration,
        model=model,
        temperature=temperature,
        seeds=seeds or [0],
        top_k=top_k,
        k_grid=k_grid or [1, 5, 10, 20, 50],
        objective=objective,
        evaluator=evaluator,
        max_tokens=max_tokens,
        dry_run=dry_run,
    )
    return await run_hillclimb(session_dir=session_dir, task=hc_task, config=hc_cfg)


__all__ = ["DiscoveryResult", "run_discovery_subagent"]
