"""Outer iterative loop for the discovery sub-agent.

Mirrors ``circuitbreaker.auto_circuit_discovery.loop`` but is callable from
an autointerp ``RunHandle`` (so the Tier-2 ``discover_features`` tool can
invoke it directly inside an investigation run).

Auth: the sub-agent uses ``claude_code_sdk`` which spawns the local
``claude`` CLI. That CLI uses subscription-based OAuth credentials stored
under ``~/.claude/`` — no ``ANTHROPIC_API_KEY`` is required for users on
Claude Pro / Max / Team subscriptions.

Two run modes:

- ``dry_run=True`` — no SDK, no LLM. Drives one iteration deterministically
  by copying the template to ``algorithm_v1.py`` and running ``evaluate.py``.
  Used by tests and by ``autointerp validate --spec ... --check-discovery``.
- ``dry_run=False`` — spawns ``claude_code_sdk.query`` per iteration.
  Requires ``pip install claude-code-sdk`` and the ``claude`` CLI logged in.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .session import update_session_files
from .system_prompt import build_discovery_system_prompt, template_path_for_substrate


SCRATCHPAD_FILE = "scratchpad.md"
EXPERIMENTS_FILE = "experiments.jsonl"
MEMORY_FILE = "memory_summary.md"
LEADERBOARD_FILE = "leaderboard.md"
LOG_FILE = "loop_log.jsonl"
STATUS_NEXT = "[NEXT]"
STATUS_DONE = "[DONE]"

_REPO_DIR = Path(__file__).resolve().parents[5]  # repo root (src/autointerp/...)


@dataclass
class DiscoveryResult:
    """Final hand-back from the sub-agent."""

    session_dir: Path
    best_candidate: str | None
    best_summary: dict[str, Any] | None
    iterations_run: int
    terminated_by: str  # "done", "max_iters", "timeout", "error"
    cost_usd: float = 0.0
    log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IterationResult:
    text: str
    turns: int
    tool_calls: int
    cost_usd: float
    usage: dict[str, Any]
    transcript: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Session bootstrapping
# ---------------------------------------------------------------------------


def _algorithm_template_path(
    substrate: str | None = None, *, real: bool = True
) -> Path:
    """Resolve the algorithm template to seed the session with.

    For real (non-dry-run) discovery sessions, returns the substrate-
    appropriate baseline so the sub-agent has a working starting point
    that exercises real model interventions. For dry-run sessions,
    returns the generic stub so torch / autointerp.tools imports stay
    deferred to actual run time.
    """
    if real and substrate is not None:
        try:
            return template_path_for_substrate(substrate)
        except ValueError:
            pass
    return Path(__file__).resolve().parent / "algorithm_template.py"


def init_session(
    session_dir: Path,
    *,
    task: str,
    reward_metric: str,
    substrate: str | None = None,
    real: bool = True,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "results").mkdir(parents=True, exist_ok=True)

    scratchpad = session_dir / SCRATCHPAD_FILE
    if not scratchpad.exists():
        scratchpad.write_text(
            "# Scratchpad\n\n"
            "## Task\n"
            f"{task}\n\n"
            "## Reward\n"
            f"Optimization target: `{reward_metric}` (higher is better).\n\n"
            "## Status\n"
            f"{STATUS_NEXT} Copy `algorithm_template.py` to `algorithm_v1.py`, "
            "run the evaluator, log the result, then propose iteration 2.\n"
        )

    experiments = session_dir / EXPERIMENTS_FILE
    if not experiments.exists():
        experiments.write_text("")

    template_dst = session_dir / "algorithm_template.py"
    if not template_dst.exists():
        template_dst.write_text(
            _algorithm_template_path(substrate, real=real).read_text()
        )

    update_session_files(session_dir)


def _read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text()


def check_done(scratchpad_text: str) -> bool:
    """Task is done iff the LAST status marker in scratchpad is ``[DONE]``."""
    last_next = scratchpad_text.rfind(STATUS_NEXT)
    last_done = scratchpad_text.rfind(STATUS_DONE)
    if last_done == -1:
        return False
    if last_next == -1:
        return True
    return last_done > last_next


def _candidate_name(n: int) -> str:
    return f"algorithm_v{n}"


def _build_results_summary(session_dir: Path) -> str:
    results_dir = session_dir / "results"
    if not results_dir.exists():
        return "No results yet."
    rows: list[tuple[str, float, str]] = []
    for cand_dir in sorted(results_dir.iterdir()):
        sp = cand_dir / "summary.json"
        if not sp.exists():
            continue
        try:
            d = json.loads(sp.read_text())
            obj = float(d.get("objective_value", 0.0))
            sub = d.get("subscores", {})
            sub_str = ", ".join(
                f"{k}={float(v):.4f}" for k, v in sub.items()
            )
            rows.append((cand_dir.name, obj, sub_str))
        except Exception:
            continue
    if not rows:
        return "No results yet."
    rows.sort(key=lambda r: r[1], reverse=True)
    lines = [f"  {name}: objective={obj:.4f}  ({sub})" for name, obj, sub in rows]
    best = rows[0]
    return (
        "Evaluated candidates (DO NOT re-read these files — use this summary):\n"
        + "\n".join(lines)
        + f"\nBEST: {best[0]} objective={best[1]:.4f}"
    )


def _build_iteration_prompt(
    *,
    task: str,
    reward_metric: str,
    iteration: int,
    max_iterations: int,
    session_dir: Path,
    dry_run: bool,
    eval_extra: str = "",
) -> str:
    candidate = _candidate_name(iteration)
    scratchpad = _read_text(session_dir / SCRATCHPAD_FILE)
    memory = _read_text(session_dir / MEMORY_FILE, "No research memory yet.")
    results_summary = _build_results_summary(session_dir)
    evaluate_script = (
        Path(__file__).resolve().parent / "evaluate.py"
    )
    dry_flag = " --dry-run" if dry_run else ""
    extra = f" {eval_extra}" if eval_extra else ""
    eval_cmd = (
        f"python {evaluate_script} --session-dir . "
        f"--algorithm ./{candidate}.py --candidate-name {candidate}"
        f"{dry_flag}{extra}"
    )
    return (
        f"## Iteration {iteration}/{max_iterations}\n\n"
        f"Task: {task}\n"
        f"Reward: `{reward_metric}` (maximize)\n\n"
        f"Session dir: {session_dir}\n"
        f"Repository root: {_REPO_DIR}\n\n"
        "Important local commands:\n"
        f"- Evaluate a candidate:\n  {eval_cmd}\n\n"
        f"## Pre-computed Results Summary\n```\n{results_summary}\n```\n\n"
        "**IMPORTANT**: The summary above is complete — do NOT re-read "
        "harness_result.json files.\n"
        f"Your job this iteration: write ONE new {candidate}.py and evaluate it. "
        "Start immediately.\n\n"
        f"## Research Memory\n```\n{memory}\n```\n\n"
        f"## Current Scratchpad\n```\n{scratchpad}\n```\n"
    )


def _append_log(
    session_dir: Path,
    iteration: int,
    result: IterationResult,
    duration_s: float,
    *,
    model: str,
) -> None:
    if result.transcript:
        tpath = session_dir / f"transcript_iter{iteration:02d}.jsonl"
        with tpath.open("w") as fh:
            for ev in result.transcript:
                fh.write(json.dumps(ev) + "\n")
    entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(duration_s, 1),
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "cost_usd": result.cost_usd,
        "usage": result.usage,
        "model": model,
        "result_preview": result.text[:500],
    }
    with (session_dir / LOG_FILE).open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Dry-run path (no SDK, no LLM, deterministic)
# ---------------------------------------------------------------------------


def _dry_run_iteration(session_dir: Path, iteration: int) -> IterationResult:
    """Deterministic single iteration: copy template, evaluate it, log."""
    candidate = _candidate_name(iteration)
    candidate_path = session_dir / f"{candidate}.py"
    if not candidate_path.exists():
        shutil.copy(session_dir / "algorithm_template.py", candidate_path)
    evaluate_script = Path(__file__).resolve().parent / "evaluate.py"
    cmd = [
        sys.executable, str(evaluate_script),
        "--session-dir", str(session_dir),
        "--algorithm", f"./{candidate}.py",
        "--candidate-name", candidate,
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    # Append a one-line experiment record so the leaderboard / memory render
    # consistently across iterations.
    summary_path = session_dir / "results" / candidate / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            with (session_dir / EXPERIMENTS_FILE).open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "id": candidate,
                            "iteration": iteration,
                            "hypothesis": "dry-run baseline",
                            "result": summary.get("subscores", {}),
                            "conclusion": (
                                f"objective={summary.get('objective_value', 0.0):.4f}"
                            ),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
    # Mark scratchpad [DONE] so check_done() terminates the loop after one
    # iteration. Tests want a single deterministic pass.
    sp = session_dir / SCRATCHPAD_FILE
    sp.write_text(sp.read_text() + f"\n{STATUS_DONE} dry-run completed iteration {iteration}.\n")
    return IterationResult(
        text=text, turns=1, tool_calls=2, cost_usd=0.0, usage={}, transcript=[]
    )


# ---------------------------------------------------------------------------
# Live path (claude_code_sdk)
# ---------------------------------------------------------------------------


def _load_sdk():
    """Try the renamed ``claude_agent_sdk`` first, fall back to legacy
    ``claude_code_sdk``.

    The two share the public shape we use; only the options class name
    differs (``ClaudeAgentOptions`` vs ``ClaudeCodeOptions``). We expose
    it under the legacy key ``ClaudeCodeOptions`` for backwards
    compatibility — callers should treat it as opaque.

    The pinned ``claude-code-sdk==0.0.25`` raises
    ``MessageParseError('Unknown message type: rate_limit_event')`` against
    current ``claude`` CLIs, so the newer SDK is strongly preferred.
    """
    try:
        from claude_agent_sdk import (  # type: ignore
            AssistantMessage,
            ClaudeAgentOptions as _Options,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            query,
        )
    except ImportError:
        try:
            from claude_code_sdk import (  # type: ignore
                AssistantMessage,
                ClaudeCodeOptions as _Options,
                ResultMessage,
                SystemMessage,
                TextBlock,
                ThinkingBlock,
                ToolUseBlock,
                query,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Discovery sub-agent (live mode) requires `claude-agent-sdk` "
                "(preferred) or the legacy `claude-code-sdk`. Install with "
                "`pip install claude-agent-sdk` and ensure the `claude` CLI "
                "is logged in (`claude /login`)."
            ) from exc
    return {
        "AssistantMessage": AssistantMessage,
        "ClaudeCodeOptions": _Options,
        "ResultMessage": ResultMessage,
        "SystemMessage": SystemMessage,
        "TextBlock": TextBlock,
        "ThinkingBlock": ThinkingBlock,
        "ToolUseBlock": ToolUseBlock,
        "query": query,
    }


async def _run_iteration_live(
    *,
    prompt: str,
    session_dir: Path,
    system_prompt: str,
    model: str,
    max_turns: int,
    permission_mode: str,
    verbose: bool,
) -> IterationResult:
    import os
    sdk = _load_sdk()
    # Forward auth-relevant env vars to the inner ``claude`` CLI subprocess.
    # The SDK doesn't auto-inherit ANTHROPIC_API_KEY, so a process running
    # under subscription auth sees no API key and falls back to
    # ~/.claude/.credentials.json. Pass explicit env so the caller can pin
    # which auth mode the sub-agent uses (and which account is billed).
    env = dict(os.environ)
    options = sdk["ClaudeCodeOptions"](
        system_prompt=system_prompt,
        model=model,
        max_turns=max_turns,
        permission_mode=permission_mode,
        cwd=str(session_dir),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        env=env,
    )
    text = ""
    turns = 0
    tools = 0
    cost = 0.0
    usage: dict[str, Any] = {}
    transcript: list[dict[str, Any]] = []
    try:
        async for message in sdk["query"](prompt=prompt, options=options):
            if isinstance(message, sdk["AssistantMessage"]):
                turns += 1
                blocks: list[dict[str, Any]] = []
                for block in message.content:
                    if isinstance(block, sdk["TextBlock"]):
                        text += block.text + "\n"
                        if verbose:
                            print(block.text)
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, sdk["ThinkingBlock"]):
                        blocks.append({"type": "thinking", "thinking": block.thinking})
                    elif isinstance(block, sdk["ToolUseBlock"]):
                        tools += 1
                        blocks.append(
                            {"type": "tool_use", "name": block.name, "input": block.input}
                        )
                transcript.append({"turn": turns, "role": "assistant", "blocks": blocks})
            elif isinstance(message, sdk["ResultMessage"]):
                cost = message.total_cost_usd or 0.0
                usage = message.usage or {}
                transcript.append(
                    {
                        "turn": turns,
                        "role": "result",
                        "cost_usd": cost,
                        "usage": usage,
                        "is_error": message.is_error,
                    }
                )
            elif isinstance(message, sdk["SystemMessage"]):
                transcript.append(
                    {"turn": turns, "role": "system", "subtype": message.subtype}
                )
    except Exception as exc:
        if turns == 0:
            raise
        text += f"\n[sdk error after {turns} turns: {exc}]"
    return IterationResult(
        text=text, turns=turns, tool_calls=tools, cost_usd=cost,
        usage=usage, transcript=transcript,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _best_candidate(session_dir: Path) -> tuple[str | None, dict[str, Any] | None]:
    results_dir = session_dir / "results"
    if not results_dir.exists():
        return None, None
    best_name: str | None = None
    best_summary: dict[str, Any] | None = None
    best_obj = -float("inf")
    for cand_dir in results_dir.iterdir():
        sp = cand_dir / "summary.json"
        if not sp.exists():
            continue
        try:
            payload = json.loads(sp.read_text())
            obj = float(payload.get("objective_value", 0.0))
            if obj > best_obj:
                best_obj = obj
                best_name = cand_dir.name
                best_summary = payload
        except Exception:
            continue
    return best_name, best_summary


def run_discovery_subagent(
    *,
    session_dir: Path,
    task: str,
    reward_metric: str,
    reward_description: str,
    substrate: str | None = None,
    component_kinds: list[str] | None = None,
    decomposition: str | None = None,
    benchmark_id: str | None = None,
    model_id: str | None = None,
    n_pairs: int = 30,
    k_grid: list[int] | None = None,
    seed: int = 42,
    split: str = "dev",
    objective: str | None = None,
    max_iterations: int = 10,
    max_turns_per_iteration: int = 20,
    iteration_timeout_s: float = 1200.0,
    model: str = "claude-sonnet-4-5",
    permission_mode: str = "bypassPermissions",
    dry_run: bool = False,
    verbose: bool = False,
) -> DiscoveryResult:
    """Run the discovery sub-agent loop end-to-end.

    Parameters
    ----------
    session_dir:
        Where to scaffold the discovery session. Must be inside the parent
        run's directory (the Tier-2 tool resolves this from the
        ``RunHandle``).
    task:
        Free-text task description for the sub-agent's scratchpad.
    reward_metric:
        Name of the reward metric (string, e.g. ``"combined_auc_k"``).
        Surfaces in the system prompt and is what the harness reports.
    reward_description:
        Short paragraph describing the reward, pulled from
        ``metrics/<name>.md`` by the caller.
    max_iterations / max_turns_per_iteration / iteration_timeout_s:
        Budget knobs. Defaults match the reference loop.
    model:
        Claude model name (default ``claude-sonnet-4-5``).
    permission_mode:
        Forwarded to ``ClaudeCodeOptions``. Default
        ``bypassPermissions`` since the sub-agent only writes inside its
        own session dir; tighten if you want per-tool approval.
    dry_run:
        If True, skip the SDK and run the deterministic stub iteration
        (copies template, evaluates it once). Lets us round-trip the
        whole loop without GPUs or subscription auth.
    verbose:
        Print per-turn text blocks live to stdout.
    """
    return _run_loop(
        session_dir=session_dir,
        task=task,
        reward_metric=reward_metric,
        reward_description=reward_description,
        substrate=substrate,
        component_kinds=component_kinds,
        decomposition=decomposition,
        benchmark_id=benchmark_id,
        model_id=model_id,
        n_pairs=n_pairs,
        k_grid=k_grid,
        seed=seed,
        split=split,
        objective=objective,
        max_iterations=max_iterations,
        max_turns_per_iteration=max_turns_per_iteration,
        iteration_timeout_s=iteration_timeout_s,
        model=model,
        permission_mode=permission_mode,
        dry_run=dry_run,
        verbose=verbose,
    )


def _setup_session_and_prompt(
    *,
    session_dir: Path,
    task: str,
    reward_metric: str,
    reward_description: str,
    substrate: str | None,
    component_kinds: list[str] | None,
    decomposition: str | None,
    benchmark_id: str | None,
    model_id: str | None,
    n_pairs: int,
    k_grid: list[int] | None,
    seed: int,
    split: str,
    objective: str | None,
    dry_run: bool,
) -> tuple[Path, str, str]:
    """Common setup: scaffold session, build system prompt, build eval-extra."""
    session_dir = Path(session_dir).expanduser().resolve()
    init_session(
        session_dir,
        task=task,
        reward_metric=reward_metric,
        substrate=substrate,
        real=not dry_run,
    )
    system_prompt = build_discovery_system_prompt(
        reward_metric=reward_metric,
        reward_description=reward_description,
        substrate=substrate or "components",
        component_kinds=component_kinds,
        decomposition=decomposition,
    )
    eval_extra_parts: list[str] = []
    if not dry_run:
        if substrate is not None:
            eval_extra_parts.append(f"--substrate {substrate}")
        if substrate == "components" and component_kinds:
            eval_extra_parts.append(f"--component-kind {component_kinds[0]}")
        if substrate == "features" and decomposition:
            eval_extra_parts.append(f"--decomposition {decomposition}")
        if benchmark_id:
            eval_extra_parts.append(f"--benchmark {benchmark_id}")
        if model_id:
            eval_extra_parts.append(f"--model-id {model_id}")
        if k_grid:
            eval_extra_parts.append(
                "--k-values " + ",".join(str(int(k)) for k in k_grid)
            )
        eval_extra_parts.append(f"--n-pairs {int(n_pairs)}")
        eval_extra_parts.append(f"--seed {int(seed)}")
        eval_extra_parts.append(f"--split {split}")
        if objective:
            eval_extra_parts.append(f"--objective {objective}")
    return session_dir, system_prompt, " ".join(eval_extra_parts)


def _run_loop(
    *,
    session_dir: Path,
    task: str,
    reward_metric: str,
    reward_description: str,
    substrate: str | None,
    component_kinds: list[str] | None,
    decomposition: str | None,
    benchmark_id: str | None,
    model_id: str | None,
    n_pairs: int,
    k_grid: list[int] | None,
    seed: int,
    split: str,
    objective: str | None,
    max_iterations: int,
    max_turns_per_iteration: int,
    iteration_timeout_s: float,
    model: str,
    permission_mode: str,
    dry_run: bool,
    verbose: bool,
) -> DiscoveryResult:
    """Sync entry point. Detects a running event loop and routes around the
    nested-asyncio bug:

    - When called from sync code (no running loop): uses ``asyncio.run`` for
      each live iteration.
    - When called from inside a running loop (e.g. the master agent's async
      handler): runs each live iteration in a *separate thread* so the
      thread can have its own event loop without interfering with the
      caller's. ``asyncio.run`` cannot be invoked from a running loop, and
      naive ``loop.run_until_complete`` would block the caller's loop.
    """
    session_dir, system_prompt, eval_extra = _setup_session_and_prompt(
        session_dir=session_dir, task=task, reward_metric=reward_metric,
        reward_description=reward_description, substrate=substrate,
        component_kinds=component_kinds, decomposition=decomposition,
        benchmark_id=benchmark_id, model_id=model_id, n_pairs=n_pairs,
        k_grid=k_grid, seed=seed, split=split, objective=objective,
        dry_run=dry_run,
    )

    # Pick an iteration runner that's safe to call from this context.
    try:
        asyncio.get_running_loop()
        nested = True
    except RuntimeError:
        nested = False

    if nested:
        # Run each live iteration in a worker thread so the new event loop
        # is fully isolated from the caller's. This is the fix for
        # ``RuntimeError: asyncio.run() cannot be called from a running
        # event loop`` when the master agent's async handler invokes us.
        import concurrent.futures as _cf

        def _run_one_live(prompt: str) -> IterationResult:
            return asyncio.run(
                asyncio.wait_for(
                    _run_iteration_live(
                        prompt=prompt, session_dir=session_dir,
                        system_prompt=system_prompt, model=model,
                        max_turns=max_turns_per_iteration,
                        permission_mode=permission_mode, verbose=verbose,
                    ),
                    timeout=iteration_timeout_s,
                )
            )

        def _run_iteration(prompt: str) -> IterationResult:
            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(_run_one_live, prompt).result()
    else:
        def _run_iteration(prompt: str) -> IterationResult:
            return asyncio.run(
                asyncio.wait_for(
                    _run_iteration_live(
                        prompt=prompt, session_dir=session_dir,
                        system_prompt=system_prompt, model=model,
                        max_turns=max_turns_per_iteration,
                        permission_mode=permission_mode, verbose=verbose,
                    ),
                    timeout=iteration_timeout_s,
                )
            )

    log: list[dict[str, Any]] = []
    iterations_run = 0
    cost_total = 0.0
    terminated_by = "max_iters"

    for iteration in range(1, max_iterations + 1):
        scratchpad = _read_text(session_dir / SCRATCHPAD_FILE)
        if check_done(scratchpad):
            terminated_by = "done"
            break

        prompt = _build_iteration_prompt(
            task=task,
            reward_metric=reward_metric,
            iteration=iteration,
            max_iterations=max_iterations,
            session_dir=session_dir,
            dry_run=dry_run,
            eval_extra=eval_extra,
        )

        start = datetime.now()
        try:
            if dry_run:
                result = _dry_run_iteration(session_dir, iteration)
            else:
                result = _run_iteration(prompt)
        except asyncio.TimeoutError:
            log.append({"iteration": iteration, "status": "timeout"})
            terminated_by = "timeout"
            break
        except Exception as exc:
            log.append(
                {
                    "iteration": iteration,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            terminated_by = "error"
            break

        elapsed = (datetime.now() - start).total_seconds()
        _append_log(session_dir, iteration, result, elapsed, model=model)
        update_session_files(session_dir)
        cost_total += result.cost_usd
        iterations_run = iteration

        log.append(
            {
                "iteration": iteration,
                "duration_s": round(elapsed, 1),
                "turns": result.turns,
                "cost_usd": result.cost_usd,
            }
        )

        if check_done(_read_text(session_dir / SCRATCHPAD_FILE)):
            terminated_by = "done"
            break

    best_name, best_summary = _best_candidate(session_dir)
    return DiscoveryResult(
        session_dir=session_dir,
        best_candidate=best_name,
        best_summary=best_summary,
        iterations_run=iterations_run,
        terminated_by=terminated_by,
        cost_usd=cost_total,
        log=log,
    )


__all__ = [
    "DiscoveryResult",
    "IterationResult",
    "check_done",
    "init_session",
    "run_discovery_subagent",
]
