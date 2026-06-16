"""Hill-climbing outer loop for the discovery sub-agent.

Each iteration:
  1. propose a new ``algorithm_v{N}.py`` (LLM, or template for iteration 1 /
     dry-run);
  2. evaluate it via the harness subprocess → reward;
  3. accept it as the new best iff the reward strictly improves (hill-climb);
  4. stop on no-improvement for ``patience`` iterations, a perfect reward, or
     ``max_iterations``.

Consistency notes:
  - The proposal step reuses ``autointerp_agent.agent_loop._call_llm`` (the
    same LiteLLM path the rest of the runtime uses) — no extra LLM SDK.
  - ``run_discovery_subagent`` is synchronous; the async Tier-2 handler calls
    it via ``asyncio.to_thread`` so each LLM call can use ``asyncio.run``
    without colliding with the master agent's event loop.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .system_prompt import build_discovery_system_prompt

_TEMPLATE = Path(__file__).resolve().parent / "algorithm_template.py"
_HARNESS = Path(__file__).resolve().parent / "harness.py"
_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class DiscoveryResult:
    """Final hand-back to the master agent."""

    session_dir: Path
    best_candidate: str | None
    best_reward: float | None
    best_summary: dict[str, Any] | None
    iterations_run: int
    terminated_by: str  # "converged" | "max_iters" | "perfect" | "error"
    log: list[dict[str, Any]] = field(default_factory=list)


def _candidate_name(n: int) -> str:
    return f"algorithm_v{n}"


def _extract_code(text: str) -> str | None:
    m = _CODE_BLOCK.search(text or "")
    if m:
        return m.group(1).strip()
    # No fence — accept the raw text iff it defines `score`.
    if text and "def score" in text:
        return text.strip()
    return None


def _evaluate(
    session_dir: Path,
    candidate_path: Path,
    *,
    top_k: int,
    k_grid: list[int],
    objective: str,
    dry_run: bool,
    evaluator: str | None,
) -> dict[str, Any] | None:
    """Run the harness on one candidate; return its summary dict or None."""
    out_path = session_dir / "results" / f"{candidate_path.stem}.json"
    cmd = [
        sys.executable, str(_HARNESS),
        "--algorithm", str(candidate_path),
        "--output", str(out_path),
        "--top-k", str(top_k),
        "--k-values", ",".join(str(k) for k in k_grid),
        "--objective", objective,
    ]
    if dry_run:
        cmd.append("--dry-run")
    elif evaluator:
        cmd += ["--evaluator", evaluator]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text())
    except Exception:
        return None


def _propose_code(
    *,
    system_prompt: str,
    task: str,
    reward_metric: str,
    iteration: int,
    best_code: str | None,
    best_reward: float | None,
    last_code: str | None,
    last_reward: float | None,
    model: str,
    temperature: float | None,
) -> str | None:
    """One LLM call that returns a new candidate module body (or None)."""
    from autointerp_agent.agent_loop import _call_llm

    parts = [
        f"Task: {task}",
        f"Reward to maximize: `{reward_metric}`.",
        f"Iteration {iteration}.",
    ]
    if best_code is not None:
        parts.append(
            f"\nBest algorithm so far (reward={best_reward:.4f}):\n"
            f"```python\n{best_code}\n```"
        )
    if last_code is not None and last_code != best_code:
        parts.append(
            f"\nYour last attempt (reward={last_reward:.4f}) did not beat the "
            f"best:\n```python\n{last_code}\n```"
        )
    parts.append(
        "\nPropose ONE improved `score(...)` module. Reply with a single "
        "fenced Python code block only."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = asyncio.run(_call_llm(**kwargs))
        text = resp.choices[0].message.content or ""
    except Exception:
        return None
    return _extract_code(text)


def run_discovery_subagent(
    *,
    session_dir: Path,
    task: str,
    reward_metric: str = "combined_auc_k",
    reward_description: str = "",
    model: str = "anthropic/claude-sonnet-4-5",
    temperature: float | None = None,
    max_iterations: int = 8,
    patience: int = 2,
    top_k: int = 20,
    k_grid: list[int] | None = None,
    objective: str = "combined",
    evaluator: str | None = None,
    dry_run: bool = False,
) -> DiscoveryResult:
    """Run the hill-climbing discovery loop end-to-end.

    Parameters
    ----------
    session_dir:
        Where to scaffold the session (resolved under the run's discovery/ tree
        by the Tier-2 tool).
    task:
        Free-text description of what to rank / discover.
    reward_metric / reward_description:
        Reward name (default ``combined_auc_k``) and a short paragraph for the
        sub-agent's prompt.
    model / temperature:
        LiteLLM model + sampling temperature for the proposal step (ignored in
        ``dry_run``). Default model matches the runtime default.
    max_iterations / patience:
        Search budget and early-stop after ``patience`` non-improving rounds.
    top_k / k_grid / objective:
        Forwarded to the harness.
    evaluator:
        ``module:attr`` for the real (GPU) evaluator. Required when
        ``dry_run`` is False.
    dry_run:
        Skip the LLM; evaluate the deterministic template once. For tests / CI.
    """
    session_dir = Path(session_dir).expanduser().resolve()
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    template_dst = session_dir / "algorithm_template.py"
    if not template_dst.exists():
        template_dst.write_text(_TEMPLATE.read_text())

    k_grid = k_grid or [1, 5, 10, 20, 50]
    system_prompt = build_discovery_system_prompt(
        reward_metric=reward_metric, reward_description=reward_description
    )

    best_name: str | None = None
    best_reward: float | None = None
    best_summary: dict[str, Any] | None = None
    best_code: str | None = None
    last_code: str | None = None
    last_reward: float | None = None
    log: list[dict[str, Any]] = []
    no_improve = 0
    iterations_run = 0
    terminated_by = "max_iters"

    n_iters = 1 if dry_run else max_iterations
    for iteration in range(1, n_iters + 1):
        candidate = _candidate_name(iteration)
        candidate_path = session_dir / f"{candidate}.py"

        if iteration == 1 or dry_run:
            # Seed from the template baseline.
            candidate_path.write_text(template_dst.read_text())
            code = candidate_path.read_text()
        else:
            proposed = _propose_code(
                system_prompt=system_prompt,
                task=task,
                reward_metric=reward_metric,
                iteration=iteration,
                best_code=best_code,
                best_reward=best_reward,
                last_code=last_code,
                last_reward=last_reward,
                model=model,
                temperature=temperature,
            )
            if proposed is None:
                log.append({"iteration": iteration, "status": "no_proposal"})
                no_improve += 1
                if no_improve >= patience:
                    terminated_by = "converged"
                    break
                continue
            code = proposed
            candidate_path.write_text(code)

        summary = _evaluate(
            session_dir, candidate_path,
            top_k=top_k, k_grid=k_grid, objective=objective,
            dry_run=dry_run, evaluator=evaluator,
        )
        iterations_run = iteration
        if summary is None:
            log.append({"iteration": iteration, "status": "eval_failed"})
            no_improve += 1
            if no_improve >= patience:
                terminated_by = "converged"
                break
            continue

        reward = float(summary.get("objective_value", 0.0))
        last_code, last_reward = code, reward
        improved = best_reward is None or reward > best_reward + 1e-9
        log.append(
            {
                "iteration": iteration,
                "candidate": candidate,
                "reward": reward,
                "improved": improved,
            }
        )
        if improved:
            best_name, best_reward, best_summary, best_code = (
                candidate, reward, summary, code,
            )
            no_improve = 0
        else:
            no_improve += 1

        if best_reward is not None and best_reward >= 1.0 - 1e-9:
            terminated_by = "perfect"
            break
        if no_improve >= patience:
            terminated_by = "converged"
            break

    (session_dir / "loop_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in log) + ("\n" if log else "")
    )
    return DiscoveryResult(
        session_dir=session_dir,
        best_candidate=best_name,
        best_reward=best_reward,
        best_summary=best_summary,
        iterations_run=iterations_run,
        terminated_by=terminated_by,
        log=log,
    )


__all__ = ["DiscoveryResult", "run_discovery_subagent"]
