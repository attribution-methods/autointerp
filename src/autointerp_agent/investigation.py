"""CLI entry point for the investigation pipeline.

Mirrors the Stage 0 CLI shape but binds the agent loop to an approved spec
and the pipeline's Tier-2 tools.

Usage:
    python -m autointerp_agent.investigation --spec outputs/specs/<id>_rev<n>.json
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.console import Console

from autointerp.pipelines.investigation import AblationFlags, RunHandle, write_report
from autointerp.pipelines.investigation.main import build_system_prompt, prepare_run
from autointerp.pipelines.investigation.observer import RunObserver
from autointerp.pipelines.investigation.state import read_state
from autointerp.pipelines.investigation.tools import create_investigation_tools
from autointerp.utils.cost import CostTracker

from .agent_loop import run_agent_turn
from .config import DEFAULT_CONFIG_PATH, AgentConfig, load_config
from .context import ContextManager
from .skills import SkillRegistry
from .tools import ToolRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autointerp investigation pipeline")
    parser.add_argument("--spec", required=True, help="Path to an approved InvestigationSpec JSON")
    parser.add_argument("--runs-root", default="runs", help="Where to scaffold the run directory")
    parser.add_argument("--prompt", default="Begin executing the investigation.",
                        help="Initial user prompt")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Require an existing run dir (refuse to init)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Refuse to resume; require a fresh run dir")
    parser.add_argument(
        "--ablation", default="full",
        help="Discipline ablation: 'full' (default, all gates on) or "
             "comma-separated OFF codes — A=frozen-spec, B=provenance-metrics, "
             "C=split-disjointness. e.g. 'A' or 'A,C'.",
    )
    parser.add_argument(
        "--wallclock", type=int, default=0,
        help="Hard wallclock kill in seconds (0 = disabled, default). The "
             "eval driver passes 1800 for D3 parity with the C0 harness.",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="Disable live console streaming of agent turns and tool calls")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the older verbose console format (iteration headers, "
                             "tool ids, full assistant blocks). Default is the compact one-"
                             "line-per-event view; jsonl artifacts are unchanged either way.")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    resume: bool | None
    if args.resume:
        resume = True
    elif args.no_resume:
        resume = False
    else:
        resume = None

    try:
        flags = AblationFlags.parse(args.ablation)
    except ValueError as exc:
        console.print(f"[red]invalid --ablation: {exc}[/red]")
        return 2

    handle, spec, state = prepare_run(
        args.spec, runs_root=Path(args.runs_root), resume=resume, flags=flags
    )
    console.print(
        f"[bold]Run[/bold]: {handle.run_id}  → {handle.root}  "
        f"[ablation: {handle.flags.label()}]"
    )
    if state.terminal_state is not None:
        console.print(f"[yellow]Run already terminal:[/yellow] {state.terminal_state.value}")
        return 0

    config = _load_cli_config(args)
    config = config.model_copy(
        update={"system_prompt": build_system_prompt(spec, handle.run_id, handle.flags)}
    )
    registry = SkillRegistry.from_dir(config.skills_dir)
    context = ContextManager(
        skill_registry=registry,
        default_skill_names=config.default_skills,
        system_prompt=config.system_prompt,
        model_name=config.model_name,
    )

    observer = RunObserver(
        handle,
        console=None if args.quiet else console,
        verbose=args.verbose,
    )

    cost_path = handle.root / "cost.json"
    cost_tracker = CostTracker.load_or_new(cost_path, run_id=handle.run_id)

    async with ToolRouter(
        skill_registry=registry,
        auto_approve=config.auto_approve,
    ) as router:
        # Range-restrict write_file/edit_file to the agent-writable roots.
        # commit_artifact remains the only path into typed artifact dirs.
        router.writable_roots = handle.writable_roots()
        router.run_dir = handle.root
        router.scratch_dir = handle.scratch_dir
        for tool in create_investigation_tools(handle):
            router.register_tool(tool)
        try:
            turn = run_agent_turn(
                args.prompt, config, context, router,
                observer=observer, cost_tracker=cost_tracker,
            )
            if args.wallclock and args.wallclock > 0:
                answer = await asyncio.wait_for(turn, timeout=args.wallclock)
            else:
                answer = await turn
        except asyncio.TimeoutError:
            answer = (
                f"[run killed: exceeded {args.wallclock}s wallclock cap] "
                "state.json/transcript are intact for scoring."
            )
            console.print(f"[red]{answer}[/red]")
        finally:
            # Always persist the cost snapshot — partial runs are still billable.
            cost_tracker.write(cost_path)
        console.print(answer)

    console.print(cost_tracker.format_summary())

    final_state = read_state(handle.state_path)
    if final_state.terminal_state is not None:
        report_path = write_report(handle)
        console.print(f"[bold green]Report written:[/bold green] {report_path}")
    else:
        console.print("[yellow]Run still in-progress; no report written this turn.[/yellow]")
    return 0


def _load_cli_config(args: argparse.Namespace) -> AgentConfig:
    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else AgentConfig()
    updates: dict = {}
    if args.model:
        updates["model_name"] = args.model
    if args.max_iterations is not None:
        updates["max_iterations"] = args.max_iterations
    if args.auto_approve:
        updates["auto_approve"] = True
    if updates:
        config = config.model_copy(update=updates)
    return config


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())


_ = RunHandle
