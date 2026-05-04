"""Unified ``autointerp`` CLI dispatcher.

Subcommands wired here:

- ``autointerp investigate <question>``   Stage 0 → finalize → run pipeline
- ``autointerp investigate --spec FILE``  Skip Stage 0; run pipeline directly
- ``autointerp runs list``                Table of every run dir
- ``autointerp runs show <id>``           Pretty-printed report.json
- ``autointerp runs tail <id>``           Live tool/turn stream
- ``autointerp validate --spec FILE``     Dry-run: parse spec + load model

Bare ``autointerp`` (no subcommand, no spec, no question) and
``autointerp "<headless prompt>"`` keep the legacy behavior so existing
docs and shell history don't break.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .config import DEFAULT_CONFIG_PATH

LEGACY_FLAGS = {
    "--config",
    "--model",
    "--max-iterations",
    "--skills-dir",
    "--auto-approve",
    "--list-skills",
}
SUBCOMMANDS = {"investigate", "runs", "validate"}


# ---------------------------------------------------------------------------
# Subcommand: investigate
# ---------------------------------------------------------------------------


def _add_investigate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", nargs="?",
                        help='Research question, e.g. "How does the model do X?"')
    parser.add_argument("--spec",
                        help="Skip Stage 0 and run directly against this approved spec")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--max-iterations", type=int, default=500,
                        help="Investigation iteration cap (default: 500)")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="Stage 0 conversation turn cap (default: 30; 0 = unlimited)")
    parser.add_argument("--auto-approve", dest="auto_approve",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-approve risky tools (default: on for non-interactive runs)")
    parser.add_argument("--resume", action="store_true",
                        help="Require an existing run dir (refuse to init)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Refuse to resume; require a fresh run dir")
    parser.add_argument("--quiet", action="store_true")


def cmd_investigate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="autointerp investigate")
    _add_investigate_args(parser)
    args = parser.parse_args(argv)

    if not args.spec and not args.question:
        parser.error("provide a question (Stage 0) or --spec FILE (skip Stage 0)")

    if args.spec:
        return _run_pipeline(args, spec_path=args.spec)

    spec_path = _stage0_to_spec(args)
    if spec_path is None:
        Console().print("[yellow]Stage 0 ended without a finalized spec — nothing to run.[/yellow]")
        return 1
    Console().print(f"[bold]Spec finalized:[/bold] {spec_path}")
    return _run_pipeline(args, spec_path=spec_path)


def _stage0_to_spec(args: argparse.Namespace) -> Optional[str]:
    """Run a Stage 0 conversational session and return the finalized spec path.

    We watch ``outputs/specs/`` for a new ``<id>_rev<n>.json`` written during
    the session (the ``finalize_spec`` tool writes there on approval). The
    user drives the conversation; this function returns once a spec lands.
    """
    from .agent_loop import run_agent_turn
    from .cli import _load_cli_config
    from .context import ContextManager
    from .skills import SkillRegistry
    from .stage0_tools import create_stage0_tools
    from .tools import ToolRouter
    from prompt_toolkit import PromptSession

    spec_dir = Path("outputs/specs")
    spec_dir.mkdir(parents=True, exist_ok=True)
    before = _snapshot_specs(spec_dir)

    cli_args = argparse.Namespace(
        config=args.config, model=args.model, max_iterations=None,
        skills_dir=None, auto_approve=False,
    )
    config = _load_cli_config(cli_args)
    registry = SkillRegistry.from_dir(config.skills_dir)
    context = ContextManager(
        skill_registry=registry,
        default_skill_names=config.default_skills,
        system_prompt=config.system_prompt,
        model_name=config.model_name,
    )

    console = Console()
    cap = args.max_turns if args.max_turns and args.max_turns > 0 else None
    cap_msg = f"max {cap} turns" if cap else "no turn cap"
    console.print(
        f"[bold]Stage 0[/bold] — design the investigation spec ({cap_msg}). "
        "Approve via [bold]finalize_spec[/bold]; this CLI then auto-launches "
        "the investigation on the approved spec."
    )

    async def _loop() -> Optional[str]:
        async with ToolRouter(
            skill_registry=registry, auto_approve=config.auto_approve,
        ) as router:
            for tool in create_stage0_tools():
                router.register_tool(tool)
            session = PromptSession()
            initial = args.question.strip() if args.question else None
            turn = 0
            while True:
                if initial:
                    prompt = initial
                    initial = None
                else:
                    try:
                        prompt = await session.prompt_async("autointerp> ")
                    except (EOFError, KeyboardInterrupt):
                        console.print()
                        return None
                    prompt = prompt.strip()
                    if not prompt:
                        continue
                turn += 1
                answer = await run_agent_turn(prompt, config, context, router)
                console.print(answer)
                new_specs = _snapshot_specs(spec_dir) - before
                if new_specs:
                    return str(sorted(new_specs)[-1])
                if cap and turn >= cap:
                    console.print(
                        f"[yellow]Stage 0 turn cap reached ({turn}/{cap}) "
                        f"without an approved spec. Re-run with "
                        f"`--max-turns N` to extend.[/yellow]"
                    )
                    return None
                if cap and turn == max(1, int(cap * 0.8)):
                    console.print(f"[dim]({turn}/{cap} turns used)[/dim]")

    return asyncio.run(_loop())


def _snapshot_specs(spec_dir: Path) -> set[Path]:
    if not spec_dir.exists():
        return set()
    return {p for p in spec_dir.glob("*_rev*.json") if not p.name.startswith("_")}


def _run_pipeline(args: argparse.Namespace, *, spec_path: str) -> int:
    """Hand off to ``autointerp_agent.investigation`` with sane defaults."""
    pipeline_argv = [
        "--spec", spec_path,
        "--runs-root", args.runs_root,
        "--config", args.config,
        "--max-iterations", str(args.max_iterations),
    ]
    if args.model:
        pipeline_argv += ["--model", args.model]
    if args.auto_approve:
        pipeline_argv += ["--auto-approve"]
    if args.resume:
        pipeline_argv += ["--resume"]
    if args.no_resume:
        pipeline_argv += ["--no-resume"]
    if args.quiet:
        pipeline_argv += ["--quiet"]

    from .investigation import main as investigation_main
    return investigation_main(pipeline_argv)


# ---------------------------------------------------------------------------
# Subcommand: runs list / show / tail
# ---------------------------------------------------------------------------


def cmd_runs(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="autointerp runs")
    sub = parser.add_subparsers(dest="action", required=True)
    p_list = sub.add_parser("list", help="Table of every run dir")
    p_list.add_argument("--runs-root", default="runs")
    p_list.add_argument("--all", action="store_true",
                        help="Include archived/_-prefixed run dirs")
    p_show = sub.add_parser("show", help="Pretty-print a run's report.json")
    p_show.add_argument("run_id")
    p_show.add_argument("--runs-root", default="runs")
    p_tail = sub.add_parser("tail", help="Live stream of tool calls + turns")
    p_tail.add_argument("run_id")
    p_tail.add_argument("--runs-root", default="runs")
    args = parser.parse_args(argv)

    if args.action == "list":
        return _runs_list(Path(args.runs_root), include_hidden=args.all)
    if args.action == "show":
        return _runs_show(Path(args.runs_root) / args.run_id)
    if args.action == "tail":
        return _runs_tail(Path(args.runs_root) / args.run_id)
    return 2


def _runs_list(runs_root: Path, *, include_hidden: bool) -> int:
    console = Console()
    if not runs_root.exists():
        console.print(f"[yellow]No runs/ directory at {runs_root}[/yellow]")
        return 0
    rows: list[tuple] = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_") and not include_hidden:
            continue
        state = _read_json(d / "state.json")
        report = _read_json(d / "report.json")
        stage = state.get("current_stage_index") if state else None
        terminal = state.get("terminal_state") if state else None
        if report and report.get("metadata", {}).get("criteria_evaluated"):
            crits = report["metadata"]["criteria_evaluated"]
            passed = sum(1 for c in crits.values() if c.get("passed"))
            crit_str = f"{passed}/{len(crits)} ✓"
        else:
            crit_str = "—"
        started = state.get("run_started_at", "") if state else ""
        ended = state.get("run_ended_at", "") if state else ""
        wall = _wallclock(started, ended)
        rows.append((d.name, str(stage) if stage is not None else "—",
                     terminal or "running" if state else "no-state",
                     crit_str, wall))

    table = Table(title=f"Runs in {runs_root}")
    table.add_column("run_id", style="bold")
    table.add_column("stage")
    table.add_column("status")
    table.add_column("criteria")
    table.add_column("wallclock")
    for r in rows:
        table.add_row(*r)
    console.print(table)
    return 0


def _runs_show(run_dir: Path) -> int:
    console = Console()
    if not run_dir.exists():
        console.print(f"[red]No such run dir:[/red] {run_dir}")
        return 1
    report = _read_json(run_dir / "report.json")
    state = _read_json(run_dir / "state.json")
    if not report:
        console.print(f"[yellow]No report.json yet at {run_dir} — run still in progress?[/yellow]")
        if state:
            console.print(f"  current_stage_index: {state.get('current_stage_index')}")
            console.print(f"  terminal_state:      {state.get('terminal_state')}")
        return 0

    md = report.get("metadata", {})
    console.print(f"[bold]{report.get('report_id','?')}[/bold]  "
                  f"[dim]({md.get('terminal_state','?')})[/dim]")
    console.print()

    crits = md.get("criteria_evaluated", {})
    if crits:
        table = Table(title="Pre-registered success criteria")
        table.add_column("criterion")
        table.add_column("metric")
        table.add_column("comp")
        table.add_column("threshold")
        table.add_column("observed")
        table.add_column("result")
        for cid, c in crits.items():
            mark = "[green]✓ pass[/green]" if c.get("passed") else "[red]✗ fail[/red]"
            table.add_row(
                cid, str(c.get("metric")), str(c.get("comparator")),
                f"{c.get('threshold')}", f"{c.get('value'):.4g}" if isinstance(c.get('value'), (int, float)) else str(c.get('value')),
                mark,
            )
        console.print(table)

    claims = report.get("claims", [])
    if claims:
        console.print("\n[bold]Claims[/bold]")
        for c in claims:
            console.print(f"  • {c}")

    findings_dir = run_dir / "findings"
    if findings_dir.exists():
        console.print(f"\n[bold]Findings on disk[/bold]: {findings_dir}")
        for stage in sorted(findings_dir.iterdir()):
            if not stage.is_dir():
                continue
            metrics = sorted(stage.glob("*.json"))
            console.print(f"  [dim]{stage.name}:[/dim] {len(metrics)} metric(s)")

    console.print(f"\nLog:    [dim]{run_dir}/INVESTIGATION_LOG.md[/dim]")
    console.print(f"Scripts: [dim]{run_dir}/scripts/[/dim]")
    return 0


def _runs_tail(run_dir: Path) -> int:
    """Tail tool_invocations.jsonl + assistant_turns.jsonl interleaved."""
    console = Console()
    if not run_dir.exists():
        console.print(f"[red]No such run dir:[/red] {run_dir}")
        return 1
    paths = {
        "tool": run_dir / "tool_invocations.jsonl",
        "turn": run_dir / "assistant_turns.jsonl",
    }
    cursors = {k: 0 for k in paths}
    console.print(f"[bold]Tailing[/bold] {run_dir}  (Ctrl-C to stop)\n")
    state_path = run_dir / "state.json"
    try:
        while True:
            for kind, p in paths.items():
                if not p.exists():
                    continue
                with p.open() as f:
                    f.seek(cursors[kind])
                    for line in f:
                        line = line.rstrip()
                        if not line:
                            continue
                        _print_stream_line(console, kind, line)
                    cursors[kind] = f.tell()
            state = _read_json(state_path)
            if state and state.get("terminal_state"):
                console.print(
                    f"\n[bold]Run finished:[/bold] {state['terminal_state']}"
                )
                return 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print()
        return 0


def _print_stream_line(console: Console, kind: str, line: str) -> None:
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        console.print(line)
        return
    if kind == "tool":
        ok = "[green]✓[/green]" if r.get("ok") else "[red]✗[/red]"
        prev = (r.get("output_preview") or "").replace("\n", " ")[:120]
        console.print(
            f"  {ok} it{r.get('iteration',0):03d} {r.get('tool',''):<22} {prev}"
        )
    else:
        text = (r.get("text") or "").strip()
        if text:
            console.print(f"[bold cyan]assistant[/bold cyan] it{r.get('iteration',0):03d}: {text[:200]}")


# ---------------------------------------------------------------------------
# Subcommand: validate (dry-run)
# ---------------------------------------------------------------------------


def cmd_validate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="autointerp validate")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--load-model", action="store_true",
                        help="Also try loading the spec's model (slower).")
    args = parser.parse_args(argv)

    console = Console()
    spec_path = Path(args.spec)
    if not spec_path.exists():
        console.print(f"[red]Spec not found:[/red] {spec_path}")
        return 1

    from autointerp.spec import InvestigationSpec
    try:
        spec = InvestigationSpec.model_validate_json(spec_path.read_text())
    except Exception as exc:
        console.print(f"[red]Spec failed to parse:[/red] {exc}")
        return 1
    console.print(f"[green]✓[/green] spec parses: {spec.spec_id} rev {spec.revision}")
    console.print(f"  question: {spec.question}")
    console.print(f"  model:    {spec.model.model_id}")
    console.print(f"  stages:   {' → '.join(s.stage for s in spec.stages)}")
    console.print(f"  criteria: {len(spec.success_criteria)} pre-registered")

    from autointerp.pipelines.investigation.metrics import REGISTRY
    from autointerp.spec import MetricName
    missing = []
    custom_names = []
    for crit in spec.success_criteria:
        if crit.metric == MetricName.CUSTOM:
            if crit.custom_metric_def is None:
                missing.append("custom (no definition)")
            else:
                custom_names.append(crit.custom_metric_def.name)
        elif crit.metric not in REGISTRY:
            missing.append(str(crit.metric))
    if missing:
        console.print(f"[red]✗[/red] unknown metrics referenced: {missing}")
        return 1
    if custom_names:
        console.print(
            f"[green]✓[/green] all criterion metrics are registered "
            f"(custom: {custom_names})"
        )
    else:
        console.print(f"[green]✓[/green] all criterion metrics are registered")

    if args.load_model:
        from autointerp.tools.model import load_model
        try:
            load_model(spec.model.model_id)
        except Exception as exc:
            console.print(f"[red]✗[/red] model load failed: {exc}")
            return 1
        console.print(f"[green]✓[/green] model loads")

    console.print("\n[bold green]Spec is runnable.[/bold green]")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _wallclock(started: str, ended: str) -> str:
    if not started:
        return "—"
    try:
        from datetime import datetime
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        e = (datetime.fromisoformat(ended.replace("Z", "+00:00"))
             if ended else datetime.now(s.tzinfo))
        delta = e - s
        secs = int(delta.total_seconds())
        return f"{secs // 60}m{secs % 60:02d}s"
    except Exception:
        return started[:16]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


TOP_LEVEL_HELP = """\
autointerp — agentic mechanistic interpretability

USAGE
  autointerp investigate <question>           Stage 0 + run pipeline (one shot)
  autointerp investigate --spec FILE          Skip Stage 0; run an approved spec
  autointerp runs list [--all]                Table of run dirs and status
  autointerp runs show <run_id>               Pretty-printed report.json
  autointerp runs tail <run_id>               Live stream of a running job
  autointerp validate --spec FILE             Dry-run: parse spec + metric check
  autointerp                                  Interactive Stage 0 (REPL)
  autointerp "<headless prompt>"              One-shot agent turn

Run any subcommand with -h for its options. See QUICKSTART.md for the
end-to-end walkthrough.
"""


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        sub, rest = argv[0], argv[1:]
        if sub == "investigate":
            return cmd_investigate(rest)
        if sub == "runs":
            return cmd_runs(rest)
        if sub == "validate":
            return cmd_validate(rest)
    if argv and argv[0] in {"-h", "--help", "help"}:
        # Only print the top-level help when there's no other flag stream that
        # the legacy CLI also recognizes (e.g. --list-skills is legacy-only).
        if not any(a in LEGACY_FLAGS for a in argv[1:]):
            print(TOP_LEVEL_HELP)
            return 0
    # Back-compat: fall through to legacy CLI (interactive Stage 0 / headless).
    from .cli import main as legacy_main
    return legacy_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
