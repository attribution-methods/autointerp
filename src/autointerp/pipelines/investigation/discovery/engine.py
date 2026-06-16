"""Generic hill-climbing search engine.

Optimizes any artifact against any reward (see ``task.HillClimbTask``). The
loop is task-agnostic; ``discovery.run_discovery_subagent`` is a thin wrapper.

Each round:
  1. pick parents from the archive (best + diverse) — or the template seed;
  2. propose ``n_subagents`` candidates concurrently (single or agentic mode);
  3. evaluate each via the harness subprocess (concurrently);
  4. merge results into a top-K archive (population);
  5. stop on no-improvement (``patience``), perfect reward, exhausted token
     budget, or ``max_iterations``.

After the search, the best candidate is optionally re-evaluated across
``seeds`` to report reward variance (fluke screening).

Async by design: proposals use the runtime's LiteLLM path and run under the
caller's event loop (the Tier-2 handler awaits this directly), so no nested
``asyncio.run`` gymnastics.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .proposer import propose_agentic, propose_single
from .system_prompt import build_hillclimb_system_prompt
from .task import HillClimbConfig, HillClimbResult, HillClimbTask

_HARNESS = Path(__file__).resolve().parent / "harness.py"
ARCHIVE_FILE = "archive.jsonl"
LOG_FILE = "loop_log.jsonl"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_sync(
    *,
    candidate_path: Path,
    out_path: Path,
    top_k: int,
    k_grid: list[int],
    objective: str,
    seed: int,
    dry_run: bool,
    evaluator: str | None,
) -> dict[str, Any] | None:
    cmd = [
        sys.executable, str(_HARNESS),
        "--algorithm", str(candidate_path),
        "--output", str(out_path),
        "--top-k", str(top_k),
        "--k-values", ",".join(str(k) for k in k_grid),
        "--objective", objective,
        "--seed", str(seed),
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


async def _evaluate(
    session_dir: Path, candidate: str, code: str, cfg: HillClimbConfig
) -> dict[str, Any] | None:
    out_path = session_dir / "results" / f"{candidate}.json"
    seed = cfg.seeds[0] if cfg.seeds else 0
    return await asyncio.to_thread(
        _evaluate_sync,
        candidate_path=session_dir / f"{candidate}.py",
        out_path=out_path,
        top_k=cfg.top_k,
        k_grid=cfg.k_grid,
        objective=cfg.objective,
        seed=seed,
        dry_run=cfg.dry_run,
        evaluator=cfg.evaluator,
    )


# ---------------------------------------------------------------------------
# Archive (population) + memory
# ---------------------------------------------------------------------------


def _load_archive(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / ARCHIVE_FILE
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _save_archive(session_dir: Path, archive: list[dict[str, Any]]) -> None:
    (session_dir / ARCHIVE_FILE).write_text(
        "".join(json.dumps(e) + "\n" for e in archive)
    )


def _merge(
    archive: list[dict[str, Any]], record: dict[str, Any], archive_size: int
) -> list[dict[str, Any]]:
    archive = archive + [record]
    archive.sort(key=lambda e: e.get("reward", -1.0), reverse=True)
    return archive[:archive_size]


def _pick_parents(archive: list[dict[str, Any]], n: int) -> list[dict[str, Any] | None]:
    """n parents for this round: best + diverse runners-up; None => template."""
    if not archive:
        return [None] * n
    parents: list[dict[str, Any] | None] = []
    for i in range(n):
        parents.append(archive[i % len(archive)])
    return parents


def _trail_block(log: list[dict[str, Any]]) -> str:
    if not log:
        return "No attempts yet."
    rows = []
    for e in log:
        cand = e.get("candidate", "?")
        if "reward" in e and e["reward"] is not None:
            rows.append(f"  {cand}: reward={e['reward']:.4f}")
        else:
            rows.append(f"  {cand}: {e.get('status', 'failed')}")
    return "\n".join(rows)


def _archive_block(archive: list[dict[str, Any]]) -> str:
    if not archive:
        return "Archive empty — you are improving the baseline template."
    parts = []
    for e in archive:
        parts.append(
            f"### {e['candidate']} (reward={e.get('reward', 0.0):.4f})\n"
            f"```python\n{e.get('code', '')}\n```"
        )
    return "\n\n".join(parts)


def _build_user_prompt(
    *,
    task: HillClimbTask,
    parent: dict[str, Any] | None,
    archive: list[dict[str, Any]],
    log: list[dict[str, Any]],
    iteration: int,
) -> str:
    if parent is None:
        parent_block = (
            "Parent: the baseline template (see the archive/contract). Improve "
            "on it."
        )
    else:
        parent_block = (
            f"Parent to improve: {parent['candidate']} "
            f"(reward={parent.get('reward', 0.0):.4f}):\n"
            f"```python\n{parent.get('code', '')}\n```"
        )
    return (
        f"Task: {task.task_text}\n"
        f"Reward to maximize: `{task.reward_metric}`.\n"
        f"Round {iteration}.\n\n"
        f"{parent_block}\n\n"
        f"## Archive (best candidates so far)\n{_archive_block(archive)}\n\n"
        f"## Trail (everything tried, including failures)\n{_trail_block(log)}\n\n"
        "Propose ONE improved candidate."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


async def _propose_one(
    *,
    task: HillClimbTask,
    cfg: HillClimbConfig,
    session_dir: Path,
    candidate: str,
    system_prompt: str,
    user_prompt: str,
    cost_tracker: Any | None,
) -> str | None:
    candidate_path = session_dir / f"{candidate}{task.artifact_suffix}"
    if cfg.propose_mode == "agentic":
        return await propose_agentic(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            candidate_path=candidate_path,
            session_dir=session_dir,
            model=cfg.model,
            temperature=cfg.temperature,
            max_turns=cfg.max_turns_per_iteration,
            cost_tracker=cost_tracker,
        )
    code = await propose_single(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=cfg.model,
        temperature=cfg.temperature,
        cost_tracker=cost_tracker,
        require_token="def score" if task.artifact_suffix == ".py" else None,
    )
    if code is not None:
        candidate_path.write_text(code)
    return code


async def run_hillclimb(
    *,
    session_dir: Path,
    task: HillClimbTask,
    config: HillClimbConfig | None = None,
) -> HillClimbResult:
    """Run the generic hill-climbing search end-to-end (async)."""
    cfg = config or HillClimbConfig()
    session_dir = Path(session_dir).expanduser().resolve()
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    template_dst = session_dir / f"template{task.artifact_suffix}"
    if not template_dst.exists():
        template_dst.write_text(task.candidate_template.read_text())

    system_prompt = build_hillclimb_system_prompt(
        reward_metric=task.reward_metric,
        reward_description=task.reward_description,
        contract_instructions=task.contract_instructions,
        agentic=(cfg.propose_mode == "agentic"),
    )

    # CostTracker bridges spend into the run budget (handler reads it back).
    from autointerp.utils.cost import CostTracker
    cost_tracker = CostTracker(run_id=session_dir.name)

    archive = _load_archive(session_dir)  # resume support
    log: list[dict[str, Any]] = []
    proposals_made = sum(1 for _ in (session_dir / "results").glob("*.json"))
    no_improve = 0
    iterations_run = 0
    terminated_by = "max_iters"

    n_rounds = 1 if cfg.dry_run else cfg.max_iterations
    width = 1 if cfg.dry_run else max(1, cfg.n_subagents)

    for rnd in range(1, n_rounds + 1):
        # Budget gate (token cap for this discovery call).
        if cfg.max_tokens is not None and cost_tracker.total_tokens >= cfg.max_tokens:
            terminated_by = "budget"
            break

        iterations_run = rnd
        parents = _pick_parents(archive, width)

        # --- propose `width` candidates concurrently ---
        async def _make(parent: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal proposals_made
            idx = proposals_made
            proposals_made += 1
            candidate = f"cand_{idx:03d}"
            if cfg.dry_run:
                # Deterministic: copy the template, no LLM.
                (session_dir / f"{candidate}{task.artifact_suffix}").write_text(
                    template_dst.read_text()
                )
                code: str | None = template_dst.read_text()
            else:
                user_prompt = _build_user_prompt(
                    task=task, parent=parent, archive=archive, log=log, iteration=rnd
                )
                code = await _propose_one(
                    task=task, cfg=cfg, session_dir=session_dir, candidate=candidate,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    cost_tracker=cost_tracker,
                )
            if code is None:
                return {"candidate": candidate, "status": "no_proposal"}
            summary = await _evaluate(session_dir, candidate, code, cfg)
            if summary is None:
                return {"candidate": candidate, "status": "eval_failed"}
            return {
                "candidate": candidate,
                "reward": float(summary.get("objective_value", 0.0)),
                "summary": summary,
                "code": code,
            }

        results = await asyncio.gather(*[_make(p) for p in parents])

        round_improved = False
        for rec in results:
            if "reward" not in rec:
                log.append({"round": rnd, **rec})
                continue
            log.append(
                {"round": rnd, "candidate": rec["candidate"], "reward": rec["reward"]}
            )
            prev_best = archive[0]["reward"] if archive else None
            archive = _merge(archive, rec, cfg.archive_size)
            if prev_best is None or archive[0]["reward"] > prev_best + 1e-9:
                round_improved = True

        _save_archive(session_dir, archive)
        no_improve = 0 if round_improved else no_improve + 1

        if archive and archive[0]["reward"] >= 1.0 - 1e-9:
            terminated_by = "perfect"
            break
        if no_improve >= cfg.patience:
            terminated_by = "converged"
            break

    # --- finalize: best + optional seed-variance ---
    best = archive[0] if archive else None
    best_reward_std = None
    if best is not None and len(cfg.seeds) > 1:
        best_reward_std = await _seed_variance(session_dir, best["candidate"], cfg)

    (session_dir / LOG_FILE).write_text(
        "".join(json.dumps(e) + "\n" for e in log)
    )
    try:
        cost_tracker.write(session_dir / "cost.json")
    except Exception:
        pass

    return HillClimbResult(
        session_dir=session_dir,
        best_candidate=best["candidate"] if best else None,
        best_reward=best["reward"] if best else None,
        best_summary=best["summary"] if best else None,
        archive=[
            {"candidate": e["candidate"], "reward": e.get("reward"),
             "summary": e.get("summary")}
            for e in archive
        ],
        iterations_run=iterations_run,
        proposals_made=proposals_made,
        terminated_by=terminated_by,
        tokens_used=cost_tracker.total_tokens,
        cost_usd=round(cost_tracker.total_cost_usd, 6),
        best_reward_std=best_reward_std,
        log=log,
    )


async def _seed_variance(
    session_dir: Path, candidate: str, cfg: HillClimbConfig
) -> float | None:
    """Re-evaluate the best candidate across seeds; return reward std (or None)."""
    rewards: list[float] = []
    for seed in cfg.seeds:
        out_path = session_dir / "results" / f"{candidate}_seed{seed}.json"
        summary = await asyncio.to_thread(
            _evaluate_sync,
            candidate_path=session_dir / f"{candidate}.py",
            out_path=out_path,
            top_k=cfg.top_k,
            k_grid=cfg.k_grid,
            objective=cfg.objective,
            seed=seed,
            dry_run=cfg.dry_run,
            evaluator=cfg.evaluator,
        )
        if summary is not None:
            rewards.append(float(summary.get("objective_value", 0.0)))
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return var ** 0.5


__all__ = ["run_hillclimb"]
