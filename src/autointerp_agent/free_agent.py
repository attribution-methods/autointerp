"""RQ1 free-agent harness (condition C0).

Runs the *same* agent runtime as the investigation pipeline (same model,
same agent loop, same RunObserver transcript schema, same range-restricted
filesystem, same hard caps) but with **none of the discipline scaffold**: no
Stage 0, no pre-registered spec, no Tier-2 gates, no skills, and a neutral
system prompt that excludes all autointerp methodology guidance.

This is the unscaffolded floor for RQ1. The only things deliberately held
constant with the scaffolded conditions are *capability* (the
``autointerp.tools.*`` helper catalog, copied verbatim from the
investigation system prompt) and the cost ceiling. See
``docs/scaffold_faithfulness_eval.md`` (decision D1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from autointerp.pipelines.investigation.main import SYSTEM_PROMPT_HEADER
from autointerp.pipelines.investigation.observer import RunObserver
from autointerp.utils.cost import CostTracker

from .agent_loop import run_agent_turn
from .config import DEFAULT_MODEL, AgentConfig
from .context import ContextManager
from .skills import SkillRegistry
from .tools import ToolRouter

# Hard caps — identical across every condition in the eval (decision D3).
DEFAULT_MAX_ITERATIONS = 80
DEFAULT_WALLCLOCK_SECONDS = 1800

# The only Tier-1 tools C0 gets: free-form work, no scaffold/Stage-0 tools.
C0_TOOLS = ("plan", "bash", "read_file", "write_file", "edit_file")

# Neutral framing: a competent researcher with a shell and the question.
# Deliberately NO methodology ("black-box first", "separate discovery from
# validation", "don't treat correlation as causal"), NO inviolable rules,
# NO scaffold "how to work". That guidance is itself a mitigation (D1).
_NEUTRAL_FRAME = """\
You are an AI researcher with full shell access to a GPU machine. Your job
is to investigate the question below and explain the underlying mechanism,
backed by whatever evidence you choose to gather.

# Question

{question}

# Environment

- Working directory: `{run_dir}`. Write scripts to `scripts/` and any data
  or intermediate files to `scratch/` — those are the only writable
  locations. Use `bash` to run them.
- A Python environment with the model tooling below is importable.

{capability_catalog}

When you are finished, give your final answer as a written explanation of
the mechanism you found and the evidence supporting it.
"""


def capability_catalog() -> str:
    """The ``autointerp.tools.*`` helper catalog, verbatim from the scaffold.

    Capability parity, not discipline: C0 must know the same helper library
    exists, or "no scaffold" is confounded with "no tool knowledge". Sliced
    out of the investigation header so the wording can never drift; the
    scaffold-specific ``current_stage`` sentence that follows it is excluded.
    """
    start = "# Helper modules already implemented — DO NOT REINVENT"
    end = "`current_stage` returns the full SKILL.md"
    i = SYSTEM_PROMPT_HEADER.find(start)
    j = SYSTEM_PROMPT_HEADER.find(end)
    if i == -1 or j == -1 or j <= i:
        raise RuntimeError(
            "capability_catalog: helper-block anchors not found in "
            "SYSTEM_PROMPT_HEADER; the slice is stale."
        )
    return SYSTEM_PROMPT_HEADER[i:j].rstrip()


def build_c0_system_prompt(question: str, run_dir: Path) -> str:
    return _NEUTRAL_FRAME.format(
        question=question.strip(),
        run_dir=run_dir,
        capability_catalog=capability_catalog(),
    )


class NeutralContext(ContextManager):
    """ContextManager with the scaffold prompt assembly stripped out.

    The base ``build_system_message`` falls back to the autointerp methodology
    prompt, lists default skills, and always appends a "use list_skills /
    read_skill" nudge. C0 must have none of that — just the neutral prompt —
    while keeping Anthropic prompt-caching (cost) intact.
    """

    def build_system_message(self) -> dict[str, Any]:
        prompt = self.system_prompt or ""
        if self._caching_enabled():
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        return {"role": "system", "content": prompt}


def _ensure_working_copy_on_path() -> None:
    """Defend against the editable-install-points-elsewhere landmine.

    Prepend this repo's ``src`` to sys.path and PYTHONPATH so the agent's
    own bash scripts import the working-copy ``autointerp`` too.
    """
    src = str(Path(__file__).resolve().parents[1])
    if src not in sys.path:
        sys.path.insert(0, src)
    existing = os.environ.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            src if not existing else src + os.pathsep + existing
        )


class _C0Handle:
    """Minimal stand-in for RunHandle — RunObserver only needs ``.root``."""

    def __init__(self, root: Path) -> None:
        self.root = root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def run_free_agent(
    *,
    question: str,
    run_dir: Path,
    model: str = DEFAULT_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    wallclock_seconds: int = DEFAULT_WALLCLOCK_SECONDS,
    console: Console | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one C0 investigation. Returns a small result dict; full transcript
    lands in ``run_dir`` in the same schema as a scaffolded run."""
    _ensure_working_copy_on_path()
    # The investigation CLI loads .env via load_config; the C0 harness has no
    # config file, so load it here too (ANTHROPIC_API_KEY lives in repo .env).
    load_dotenv(override=False)
    run_dir = Path(run_dir)
    scripts_dir = run_dir / "scripts"
    scratch_dir = run_dir / "scratch"
    for d in (run_dir, scripts_dir, scratch_dir):
        d.mkdir(parents=True, exist_ok=True)

    system_prompt = build_c0_system_prompt(question, run_dir)
    (run_dir / "system_prompt.txt").write_text(system_prompt)

    config = AgentConfig(
        model_name=model,
        max_iterations=max_iterations,
        auto_approve=True,
        default_skills=[],
        system_prompt=system_prompt,
    )
    registry = SkillRegistry({})  # no skills in C0
    context = NeutralContext(
        skill_registry=registry,
        default_skill_names=[],
        system_prompt=system_prompt,
        model_name=model,
    )
    observer = RunObserver(
        _C0Handle(run_dir), console=console, verbose=verbose
    )
    cost_path = run_dir / "cost.json"
    run_id = run_dir.name
    cost_tracker = CostTracker.load_or_new(cost_path, run_id=run_id)

    started = _now()
    answer = ""
    stopped_reason = "final"
    async with ToolRouter(skill_registry=registry, auto_approve=True) as router:
        # Prune the auto-registered builtins down to the neutral free-form
        # set — drops the Stage-0 spec tools and the skill tools.
        router.tools = {k: v for k, v in router.tools.items() if k in C0_TOOLS}
        router._aliases = {
            a: t for a, t in router._aliases.items() if t in C0_TOOLS
        }
        router.writable_roots = [scripts_dir, scratch_dir]
        router.scratch_dir = scratch_dir
        try:
            answer = await asyncio.wait_for(
                run_agent_turn(
                    "Begin your investigation.",
                    config,
                    context,
                    router,
                    observer=observer,
                    cost_tracker=cost_tracker,
                ),
                timeout=wallclock_seconds,
            )
            if answer.startswith("Stopped after max_iterations"):
                stopped_reason = "max_iterations"
        except asyncio.TimeoutError:
            stopped_reason = "wallclock_timeout"
            answer = f"[run killed: exceeded {wallclock_seconds}s wallclock cap]"
        except Exception as exc:  # noqa: BLE001 — record, don't crash the batch
            stopped_reason = "error"
            answer = f"[run errored: {type(exc).__name__}: {exc}]"
        finally:
            cost_tracker.write(cost_path)

    (run_dir / "c0_answer.md").write_text(answer)
    meta = {
        "condition": "C0",
        "run_id": run_id,
        "model": model,
        "max_iterations": max_iterations,
        "wallclock_seconds": wallclock_seconds,
        "started_at": started,
        "ended_at": _now(),
        "stopped_reason": stopped_reason,
        "answer_chars": len(answer),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _read_question(args: argparse.Namespace) -> str:
    if args.question_file:
        return Path(args.question_file).read_text()
    if args.question:
        return args.question
    raise SystemExit("provide --question or --question-file")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RQ1 free-agent harness (C0): scaffold-free investigation"
    )
    p.add_argument("--question", help="The research question (matched, blinded)")
    p.add_argument("--question-file", help="File containing the research question")
    p.add_argument("--run-dir", required=True, help="Where to write the run")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--wallclock", type=int, default=DEFAULT_WALLCLOCK_SECONDS)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = None if args.quiet else Console()
    meta = asyncio.run(
        run_free_agent(
            question=_read_question(args),
            run_dir=Path(args.run_dir),
            model=args.model,
            max_iterations=args.max_iterations,
            wallclock_seconds=args.wallclock,
            console=console,
            verbose=args.verbose,
        )
    )
    if console is not None:
        console.print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
