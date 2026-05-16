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

from autointerp.pipelines.investigation import RunHandle, write_report
from autointerp.pipelines.investigation.main import build_system_prompt, prepare_run
from autointerp.pipelines.investigation.observer import RunObserver
from autointerp.pipelines.investigation.state import read_state
from autointerp.pipelines.investigation.tools import create_investigation_tools

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

    handle, spec, state = prepare_run(
        args.spec, runs_root=Path(args.runs_root), resume=resume
    )
    console.print(f"[bold]Run[/bold]: {handle.run_id}  → {handle.root}")
    if state.terminal_state is not None:
        console.print(f"[yellow]Run already terminal:[/yellow] {state.terminal_state.value}")
        return 0

    config = _load_cli_config(args)
    config = config.model_copy(
        update={"system_prompt": build_system_prompt(spec, handle.run_id)}
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

    async with ToolRouter(
        skill_registry=registry,
        auto_approve=config.auto_approve,
    ) as router:
        for tool in create_investigation_tools(handle):
            router.register_tool(tool)
        answer = await run_agent_turn(
            args.prompt, config, context, router, observer=observer
        )
        console.print(answer)

    final_state = read_state(handle.state_path)
    if final_state.terminal_state is not None:
        report_path = write_report(handle)
        console.print(f"[bold green]Report written:[/bold green] {report_path}")
        try:
            from autointerp.pipelines.investigation.panel import write_snapshot

            snap = write_snapshot(handle.root)
            console.print(
                f"[bold green]Results panel:[/bold green] {snap}\n"
                f"  Interactive view: "
                f"[cyan]autointerp runs panel {handle.run_id} "
                f"--runs-root {handle.root.parent}[/cyan]"
            )
        except Exception as exc:  # never let the panel break a finished run
            console.print(
                f"[yellow]Results panel skipped ({type(exc).__name__}: "
                f"{exc}); run artifacts are intact.[/yellow]"
            )
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
