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


# Keep the *inline* trail bounded; older attempts stay in loop_log.jsonl.
_TRAIL_TAIL = 12


def _leaderboard(archive: list[dict[str, Any]]) -> str:
    """Compact one-line-per-candidate summary (no code)."""
    if not archive:
        return "Archive empty — you are improving the baseline template."
    return "\n".join(
        f"  {e['candidate']}: reward={e.get('reward', 0.0):.4f}" for e in archive
    )


def _insights(log: list[dict[str, Any]]) -> str:
    """Deterministic rolling summary distilled from the trail (no LLM)."""
    rewarded = [e for e in log if e.get("reward") is not None]
    failed = [e for e in log if e.get("reward") is None]
    if not rewarded:
        return f"No successful evaluations yet ({len(failed)} failed proposals)."
    best_i = max(range(len(rewarded)), key=lambda i: rewarded[i]["reward"])
    best = rewarded[best_i]
    since = len(rewarded) - 1 - best_i
    fail_kinds: dict[str, int] = {}
    for e in failed:
        fail_kinds[e.get("status", "failed")] = fail_kinds.get(e.get("status", "failed"), 0) + 1
    fail_note = (
        "; failures: " + ", ".join(f"{k}×{v}" for k, v in sorted(fail_kinds.items()))
        if fail_kinds else ""
    )
    return (
        f"best={best['reward']:.4f} ({best['candidate']}); "
        f"{since} successful attempts since best; "
        f"{len(rewarded)} evaluated, {len(failed)} failed{fail_note}."
    )


def _trail_tail(log: list[dict[str, Any]]) -> str:
    if not log:
        return "No attempts yet."
    tail = log[-_TRAIL_TAIL:]
    rows = []
    for e in tail:
        cand = e.get("candidate", "?")
        if e.get("reward") is not None:
            rows.append(f"  {cand}: reward={e['reward']:.4f}")
        else:
            rows.append(f"  {cand}: {e.get('status', 'failed')}")
    prefix = (
        f"  (+{len(log) - len(tail)} earlier attempts — see loop_log.jsonl)\n"
        if len(log) > len(tail) else ""
    )
    return prefix + "\n".join(rows)


def _inline_code_block(
    parent: dict[str, Any] | None,
    archive: list[dict[str, Any]],
    *,
    n_extra: int,
) -> str:
    """Full code for the parent + up to n_extra other top candidates (deduped)."""
    shown: list[dict[str, Any]] = []
    seen: set[str] = set()
    if parent is not None:
        shown.append(parent)
        seen.add(parent["candidate"])
    for e in archive:
        if len(shown) >= 1 + max(0, n_extra):
            break
        if e["candidate"] not in seen:
            shown.append(e)
            seen.add(e["candidate"])
    if not shown:
        return "Baseline template only (see the contract). Improve on it."
    parts = []
    for e in shown:
        role = "parent (improve this)" if e is parent else "archived"
        parts.append(
            f"### {e['candidate']} — {role} (reward={e.get('reward', 0.0):.4f})\n"
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
    n_inline: int,
    agentic: bool,
    session_dir: Path,
) -> str:
    lookup = ""
    if agentic:
        lookup = (
            "\n## Structured memory (read on demand)\n"
            f"Full code + scores for every candidate live under `{session_dir}`:\n"
            "- `archive.jsonl` — current top-K (candidate, reward, code)\n"
            "- `results/<candidate>.json` — full eval payload per candidate\n"
            "- `<candidate>.py` — each candidate's source\n"
            "Use your file tools to read any you want to study; only the parent "
            "and a couple of top candidates are inlined below to save context.\n"
        )
    return (
        f"Task: {task.task_text}\n"
        f"Reward to maximize: `{task.reward_metric}`.\n"
        f"Round {iteration}.\n\n"
        f"## Progress\n{_insights(log)}\n\n"
        f"## Leaderboard (top candidates)\n{_leaderboard(archive)}\n\n"
        f"## Recent trail\n{_trail_tail(log)}\n"
        f"{lookup}\n"
        f"## Candidate code\n{_inline_code_block(parent, archive, n_extra=n_inline)}\n\n"
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
    terminated_by = "seeded" if cfg.dry_run else "max_iters"

    async def _eval_code(candidate: str, code: str) -> dict[str, Any]:
        summary = await _evaluate(session_dir, candidate, code, cfg)
        if summary is None:
            return {"candidate": candidate, "status": "eval_failed"}
        return {
            "candidate": candidate,
            "reward": float(summary.get("objective_value", 0.0)),
            "summary": summary,
            "code": code,
        }

    # --- Seed the population with the evaluated baseline template, so the
    # search never returns worse than its starting point (baseline = floor).
    if not archive:
        seed_name = f"cand_{proposals_made:03d}"
        proposals_made += 1
        seed_code = template_dst.read_text()
        (session_dir / f"{seed_name}{task.artifact_suffix}").write_text(seed_code)
        rec = await _eval_code(seed_name, seed_code)
        log.append({"round": 0, **{k: rec[k] for k in rec if k != "summary" and k != "code"}})
        if "reward" in rec:
            archive = _merge(archive, rec, cfg.archive_size)
            _save_archive(session_dir, archive)
        iterations_run = 1

    # Dry-run stops at the seeded baseline (no LLM).
    n_rounds = 0 if cfg.dry_run else cfg.max_iterations
    width = max(1, cfg.n_subagents)

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
            user_prompt = _build_user_prompt(
                task=task, parent=parent, archive=archive, log=log, iteration=rnd,
                n_inline=cfg.archive_code_in_context,
                agentic=(cfg.propose_mode == "agentic"), session_dir=session_dir,
            )
            code = await _propose_one(
                task=task, cfg=cfg, session_dir=session_dir, candidate=candidate,
                system_prompt=system_prompt, user_prompt=user_prompt,
                cost_tracker=cost_tracker,
            )
            if code is None:
                return {"candidate": candidate, "status": "no_proposal"}
            return await _eval_code(candidate, code)

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
