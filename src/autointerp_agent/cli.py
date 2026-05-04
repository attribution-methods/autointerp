"""Command-line entry point for the autointerp agent."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from rich.console import Console

from .agent_loop import run_agent_turn
from .config import DEFAULT_CONFIG_PATH, AgentConfig, load_config
from .context import ContextManager
from .skills import SkillRegistry
from .tools import ToolRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autointerp agent")
    parser.add_argument("prompt", nargs="?", help="Headless prompt to run")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML/JSON config",
    )
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--max-iterations", type=int, help="Maximum LLM/tool loop iterations")
    parser.add_argument("--skills-dir", help="Override skill directory")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve local write/edit/risky tools",
    )
    parser.add_argument("--list-skills", action="store_true", help="List available skills and exit")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Cap on user/agent turns in interactive mode (default: 30; 0 = unlimited)",
    )
    return parser


_SPEC_DIR = Path("outputs/specs")


def _snapshot_specs() -> set[Path]:
    if not _SPEC_DIR.exists():
        return set()
    return {p for p in _SPEC_DIR.glob("*_rev*.json") if not p.name.startswith("_")}


def _auto_launch_pipeline(spec_path: str, console: Console) -> int:
    """Hand off an approved spec to the investigation pipeline.

    Mirrors the auto-launch behavior of ``autointerp investigate`` so the
    bare REPL doesn't dead-end after ``finalize_spec``.
    """
    console.print(f"[bold]Spec finalized:[/bold] {spec_path}")
    console.print("[bold]Launching investigation…[/bold]")
    pipeline_argv = [
        "--spec", spec_path,
        "--runs-root", "runs",
        "--max-iterations", "500",
        "--auto-approve",
    ]
    from .investigation import main as investigation_main
    return investigation_main(pipeline_argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_cli_config(args)
    registry = SkillRegistry.from_dir(config.skills_dir)
    console = Console()
    if args.list_skills:
        for line in registry.list_lines():
            console.print(line)
        return 0

    context = ContextManager(
        skill_registry=registry,
        default_skill_names=config.default_skills,
        system_prompt=config.system_prompt,
        model_name=config.model_name,
    )
    _SPEC_DIR.mkdir(parents=True, exist_ok=True)
    specs_before = _snapshot_specs()
    finalized_spec: str | None = None

    async with ToolRouter(
        skill_registry=registry,
        auto_approve=config.auto_approve,
        mcp_servers={
            name: server.model_dump(exclude_none=True)
            for name, server in config.mcpServers.items()
        },
    ) as router:
        if args.prompt:
            answer = await run_agent_turn(args.prompt, config, context, router)
            console.print(answer)
            new_specs = _snapshot_specs() - specs_before
            if new_specs:
                finalized_spec = str(sorted(new_specs)[-1])
        else:
            session = PromptSession()
            cap = args.max_turns if args.max_turns and args.max_turns > 0 else None
            cap_msg = f"max {cap} turns" if cap else "no turn cap"
            console.print(
                f"[bold]Autointerp[/bold] interactive mode ({cap_msg}). Ctrl-D to exit. "
                "Approving a spec via [bold]finalize_spec[/bold] auto-launches "
                "the investigation."
            )
            turn = 0
            while True:
                try:
                    prompt = await session.prompt_async("autointerp> ")
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    return 0
                prompt = prompt.strip()
                if not prompt:
                    continue
                turn += 1
                answer = await run_agent_turn(prompt, config, context, router)
                console.print(answer)
                new_specs = _snapshot_specs() - specs_before
                if new_specs:
                    finalized_spec = str(sorted(new_specs)[-1])
                    break
                if cap and turn >= cap:
                    console.print(
                        f"[yellow]Turn cap reached ({turn}/{cap}). "
                        f"Run `autointerp --max-turns N` to extend, or Ctrl-D to exit.[/yellow]"
                    )
                    return 0
                if cap and turn == max(1, int(cap * 0.8)):
                    console.print(
                        f"[dim]({turn}/{cap} turns used)[/dim]"
                    )

    if finalized_spec is not None:
        return _auto_launch_pipeline(finalized_spec, console)
    return 0


def _load_cli_config(args: argparse.Namespace) -> AgentConfig:
    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else AgentConfig()
    updates = {}
    if args.model:
        updates["model_name"] = args.model
    if args.max_iterations is not None:
        updates["max_iterations"] = args.max_iterations
    if args.skills_dir:
        updates["skills_dir"] = args.skills_dir
    if args.auto_approve:
        updates["auto_approve"] = True
    if updates:
        config = config.model_copy(update=updates)
    return config


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
