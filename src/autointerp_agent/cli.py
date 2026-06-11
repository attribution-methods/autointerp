"""Command-line entry point for the autointerp agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from .agent_loop import run_agent_turn
from .config import (
    DEFAULT_CONFIG_PATH,
    AgentConfig,
    env_default_model,
    load_config,
    load_env_files,
)
from .context import ContextManager
from .model_select import ensure_model_ready, format_llm_error
from .repl import is_first_run, mark_first_run_complete, print_banner, run_spec_repl
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
        default=0,
        help="Optional cap on interactive turns (default: unlimited — you are "
             "the limit; live cost shows in the input box status bar)",
    )
    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Pick a previous session in this project and resume it "
             "(conversation, draft plan, turns, and cost are restored)",
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


async def async_main(argv: list[str] | None = None) -> int | str:
    args = build_parser().parse_args(argv)
    config = _load_cli_config(args)
    registry = SkillRegistry.from_dir(config.skills_dir)
    console = Console()
    if args.list_skills:
        for line in registry.list_lines():
            console.print(line)
        return 0

    interactive = args.prompt is None
    if interactive:
        print_banner(console, config=config, registry=registry, first_run=is_first_run())
    ready = await ensure_model_ready(config, console, interactive=sys.stdin.isatty())
    if ready is None:
        return 1
    config = ready
    session_store = None
    resume_payload = None
    if interactive:
        mark_first_run_complete()
        from .sessions import SessionStore, pick_session

        session_store = SessionStore()
        if args.continue_session and sys.stdin.isatty():
            resume_payload = await pick_session(session_store, console)

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
            try:
                answer = await run_agent_turn(args.prompt, config, context, router)
            except Exception as exc:  # noqa: BLE001 — print, don't traceback
                console.print(f"[red]{format_llm_error(exc)}[/red]")
                return 1
            console.print(answer)
            new_specs = _snapshot_specs() - specs_before
            if new_specs:
                finalized_spec = str(sorted(new_specs)[-1])
        else:
            finalized_spec = await run_spec_repl(
                config=config,
                context=context,
                router=router,
                console=console,
                registry=registry,
                spec_dir=_SPEC_DIR,
                max_turns=args.max_turns,
                session_store=session_store,
                resume_payload=resume_payload,
            )

    # Returning a string signals main() to invoke the pipeline. We must NOT
    # call investigation_main here: it does its own asyncio.run, and nesting
    # event loops raises "asyncio.run() cannot be called from a running
    # event loop". Hand off to sync code after asyncio.run completes.
    if finalized_spec is not None:
        return finalized_spec
    return 0


def _load_cli_config(args: argparse.Namespace) -> AgentConfig:
    load_env_files()  # credentials/.env load even when no config file exists
    config_path = Path(args.config)
    config = (
        load_config(config_path)
        if config_path.exists()
        else AgentConfig(model_name=env_default_model())
    )
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
    try:
        result = asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        # Ctrl-C outside the prompt/turn handlers (e.g. during startup or
        # setup): exit quietly with the conventional code, never a traceback.
        print()
        return 130
    if isinstance(result, str):
        return _auto_launch_pipeline(result, Console())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
