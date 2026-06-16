"""CLI entry point for the investigation pipeline.

Mirrors the Stage 0 CLI shape but binds the agent loop to an approved spec
and the pipeline's Tier-2 tools.

Usage:
    python -m autointerp_agent.investigation --spec outputs/specs/<id>_rev<n>.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Awaitable, Callable

from rich import box as rich_box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autointerp.pipelines.investigation import AblationFlags, RunHandle, write_report
from autointerp.pipelines.investigation.main import build_system_prompt, prepare_run
from autointerp.pipelines.investigation.observer import RunObserver
from autointerp.pipelines.investigation.state import RunState, read_state
from autointerp.pipelines.investigation.tools import create_investigation_tools
from autointerp.spec import InvestigationSpec
from autointerp.utils.cost import CostTracker

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
from .skills import SkillRegistry
from .tools import ToolRouter

# The investigation runs autonomously to a terminal state. A single
# `run_agent_turn` returns the moment the agent emits text with no tool call
# — which a weaker model does to "ask the user" mid-run, stranding the run.
# The driver re-engages it (bounded) until the run is genuinely terminal.
MAX_CONTINUATIONS = 10

_CONTINUE_NUDGE = (
    "You stopped, but the investigation has NOT reached a terminal state. "
    "You are running autonomously — there is no human to answer questions. "
    "Do not ask for confirmation. Continue executing the plan now: run the "
    "next tool, commit the required artifacts, evaluate criteria, and call "
    "advance_stage until the run is genuinely terminal (all success criteria "
    "evaluated, a criterion fails, or you call request_spec_revision). If you "
    "are blocked on a real methodological problem, call request_spec_revision "
    "with the reason — do not just stop."
)


async def _drive_to_completion(
    handle: RunHandle,
    run_turn: Callable[[str], Awaitable[str]],
    initial_prompt: str,
    *,
    max_continuations: int = MAX_CONTINUATIONS,
) -> tuple[str, str]:
    """Re-engage the agent until the run reaches a terminal state.

    ``run_turn(prompt)`` runs one full agent turn (which itself loops over
    tool calls). Between turns we check ``state.json``: a terminal state or
    the inner max-iterations cap ends the drive; otherwise the agent stopped
    voluntarily (typically to ask a question) and we nudge it to continue,
    bounded by ``max_continuations``. Returns ``(last_answer, stop_reason)``.
    """
    prompt = initial_prompt
    answer = ""
    for _ in range(max_continuations + 1):
        answer = await run_turn(prompt)
        if read_state(handle.state_path).terminal_state is not None:
            return answer, "terminal"
        if answer.startswith("Stopped after max_iterations"):
            return answer, "max_iterations"
        prompt = _CONTINUE_NUDGE
    return answer, "continuation_cap"


def _criteria_tally(state: RunState) -> tuple[int, int]:
    """(passed, total) over evaluated criteria, tolerant of record shape."""
    records = getattr(state, "criteria_evaluated", {}) or {}
    passed = 0
    for rec in records.values():
        verdict = getattr(rec, "verdict", None)
        if verdict is not None:
            passed += 1 if str(getattr(verdict, "value", verdict)) == "pass" else 0
        elif getattr(rec, "passed", None):
            passed += 1
    return passed, len(records)


def _print_run_header(console: Console, spec: InvestigationSpec, handle: RunHandle) -> None:
    stages = " → ".join(s.stage.value for s in spec.stages)
    body = Group(
        Text.from_markup(
            f"[bold]{spec.spec_id}[/bold] rev {spec.revision}"
            f"  [dim]· {handle.flags.label()}[/dim]"
        ),
        Text(spec.question, style="dim"),
        Text(""),
        Text.from_markup(f"[dim]stages  [/dim] {stages}"),
        Text.from_markup(
            f"[dim]criteria[/dim] {len(spec.success_criteria)} pre-registered · "
            f"runs autonomously · writes to runs/{handle.run_id}/"
        ),
    )
    console.print(
        Panel(body, box=rich_box.ROUNDED, border_style="cyan",
              title="running investigation", title_align="left", padding=(0, 1))
    )


def _grounding_warning(report_path: Path | None) -> str | None:
    """The report's "no model execution detected" flag, if set — surfaced so a
    fabricated run never reads as a clean success."""
    if report_path is None or not report_path.exists():
        return None
    import json
    try:
        report = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return report.get("metadata", {}).get("grounding_warning")


def _print_run_summary(
    console: Console, handle: RunHandle, state: RunState, reason: str,
    report_path: Path | None, *, spec_arg: str,
) -> None:
    if (state.terminal_state is not None
            and state.terminal_state.value == "revision_requested"):
        # Owned by _render_inline_results, which shows a plain-language,
        # actionable panel instead of a misleading "finished" one.
        return
    if reason == "terminal" and state.terminal_state is not None:
        ts = state.terminal_state.value
        grounding = _grounding_warning(report_path)
        color = {"completed": "green", "criterion_failed": "red"}.get(ts, "yellow")
        title = "investigation finished"
        passed, total = _criteria_tally(state)
        lines = [Text.from_markup(f"[bold {color}]{ts.replace('_', ' ')}[/bold {color}]")]
        if grounding:
            # A "completed" run that never ran the model is not a success.
            color = "red"
            title = "investigation finished — results UNVERIFIED"
            lines.append(Text.from_markup(f"[bold red]⚠ {grounding}[/bold red]"))
        if total:
            lines.append(Text.from_markup(f"[dim]criteria[/dim] {passed}/{total} passed"))
        if report_path is not None:
            lines.append(Text.from_markup(f"[dim]report  [/dim] {report_path}"))
        lines.append(
            Text.from_markup(f"[dim]inspect [/dim] autointerp runs show {handle.run_id}")
        )
        console.print(
            Panel(Group(*lines), box=rich_box.ROUNDED, border_style=color,
                  title=title, title_align="left", padding=(0, 1))
        )
        return
    why = {
        "wallclock": "hit the wallclock cap",
        "continuation_cap": "the agent kept stopping without finishing",
        "max_iterations": "hit the per-turn step cap",
        "error": "ended on an error",
    }.get(reason, "did not reach a conclusion")
    console.print(
        Panel(
            Group(
                Text.from_markup(f"[yellow]Run paused[/yellow] — {why}."),
                Text.from_markup(
                    f"[dim]resume[/dim]  autointerp investigate --spec {spec_arg}"
                ),
            ),
            box=rich_box.ROUNDED, border_style="yellow",
            title="investigation paused", title_align="left", padding=(0, 1),
        )
    )


def _render_inline_results(console: Console, handle: RunHandle, state: RunState) -> None:
    """Show the run's outcome in-session — criteria verdicts, findings, and a
    revision reason if one was requested — so the user never needs a separate
    ``autointerp runs show`` to learn what happened."""
    import json

    report_path = handle.root / "report.json"
    crits: dict = {}
    claims: list = []
    grounding: str | None = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError):
            report = {}
        crits = report.get("metadata", {}).get("criteria_evaluated", {}) or {}
        claims = report.get("claims", []) or []
        grounding = report.get("metadata", {}).get("grounding_warning")

    # A loud warning ABOVE the verdicts, so passing checkmarks computed on
    # fabricated inputs can never read as a real result.
    if grounding:
        console.print(Panel(
            Text.from_markup(f"[bold red]⚠ {grounding}[/bold red]"),
            box=rich_box.ROUNDED, border_style="red",
            title="results not trustworthy", title_align="left", padding=(0, 1),
        ))

    # What the run set out to test (from the frozen spec) — so the verdicts
    # below read as evidence about a hypothesis, not bare numbers.
    question = hypothesis = ""
    registered_total = 0  # pre-registered criteria count — the honest denominator
    spec_path = handle.root / "spec.json"
    if spec_path.exists():
        try:
            sp = json.loads(spec_path.read_text())
            question = str(sp.get("question") or "").strip()
            hypothesis = str(sp.get("hypothesis") or "").strip()
            registered_total = len(sp.get("success_criteria") or [])
        except (json.JSONDecodeError, OSError):
            pass
    if crits and (question or hypothesis):
        body = []
        if question:
            body.append(Text.from_markup(f"[dim]question  [/dim] {question}"))
        if hypothesis:
            body.append(Text.from_markup(f"[dim]hypothesis[/dim] {hypothesis}"))
        console.print(Panel(
            Group(*body), box=rich_box.ROUNDED, border_style="dim",
            title="what we set out to test", title_align="left", padding=(0, 1),
        ))

    if crits:
        table = Table(
            title="Pre-registered success criteria", box=rich_box.ROUNDED,
            title_justify="left",
        )
        for col in ("criterion", "metric", "comp", "threshold", "observed", "result"):
            table.add_column(col)
        marks = {
            "pass": "[green]✓ pass[/green]",
            "fail": "[red]✗ fail[/red]",
            "inconclusive": "[yellow]? inconclusive[/yellow]",
        }
        for cid, c in crits.items():
            value = c.get("value")
            observed = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
            table.add_row(
                cid, str(c.get("metric")), str(c.get("comparator")),
                str(c.get("threshold")), observed,
                marks.get(c.get("verdict"), "[dim]?[/dim]"),
            )
        console.print(table)

    # Bottom line: tie the verdicts back to the hypothesis in plain words.
    passed = sum(1 for c in crits.values() if c.get("verdict") == "pass")
    evaluated = len(crits)
    # The denominator is what was PRE-REGISTERED, not just what got evaluated —
    # otherwise a run that tested 1 of 3 criteria and paused reads as "1/1 passed
    # → SUPPORTED", overclaiming a hypothesis whose causal half was never tested.
    total = max(registered_total, evaluated)
    ts = getattr(state.terminal_state, "value", state.terminal_state)
    # These states stop a run BEFORE its pre-registered criteria are all tested,
    # so a partial pass tally must NOT read as a final verdict (the user's run
    # paused for a revision after 1 of 3 criteria, yet reported "SUPPORTED").
    # 'completed' and 'criterion_failed' are definitive; None is only the
    # mid-flight case exercised in tests.
    incomplete = ts in {"revision_requested", "aborted", "budget_exhausted"}
    if grounding:
        console.print(
            "[bold red]Bottom line:[/bold red] the model was never actually run, so "
            "these criteria are [bold]not evidence[/bold] about the hypothesis — it was "
            "neither confirmed nor refuted. Re-run so the metrics come from a real "
            "model to actually test it."
        )
    elif incomplete and total:
        where = (
            "paused for a plan revision" if ts == "revision_requested"
            else "did not finish"
        )
        tested = (
            f"{passed} of {evaluated} passed so far" if evaluated
            else "none were tested"
        )
        console.print(
            f"[bold yellow]Bottom line:[/bold yellow] the run is [bold]INCOMPLETE[/bold] "
            f"— only {evaluated} of {total} pre-registered criteria were tested "
            f"({tested}), then it {where}. The hypothesis is [bold]not yet fully "
            f"tested[/bold]; the remaining criteria still need to run before any "
            f"verdict holds."
        )
    elif total:
        if passed == total:
            verdict = "[green]SUPPORTED[/green] by the pre-registered criteria"
        elif passed == 0:
            verdict = "[red]NOT supported[/red] by the pre-registered criteria"
        else:
            verdict = "[yellow]PARTIALLY supported[/yellow] by the pre-registered criteria"
        console.print(
            f"[bold]Bottom line:[/bold] {passed}/{total} criteria passed — "
            f"the hypothesis is {verdict}. (The agent's message above explains what "
            f"this means and how the evidence was obtained.)"
        )

    # Findings — but never present passing claims as real when the run wasn't
    # grounded; the verdicts there were computed on data the model never produced.
    if claims and not grounding:
        console.print("[bold]Findings[/bold]")
        for claim in claims:
            console.print(f"  • {claim}")
    elif claims and grounding:
        console.print(
            "[dim]Per-criterion claims are withheld here — the run was not grounded, "
            "so they are not findings.[/dim]"
        )

    req = getattr(state, "spec_revision_requested", None)
    if req is not None:
        reason = (getattr(req, "reason", None) or "").strip()
        lines = [Text.from_markup(
            "[yellow]The agent paused and asked to change the plan[/yellow] "
            "before finishing the investigation."
        )]
        if reason:
            lines.append(Text.from_markup(f"[dim]why:[/dim] {reason}"))
        lines.append(Text.from_markup(
            "[bold]What you can do[/bold] — just reply in the chat:\n"
            "  • say how to adjust the plan (e.g. \"use a simpler metric\", or "
            "\"go ahead, revise it and re-run\") and we'll re-plan and retry;\n"
            "  • or ask why it stopped, or start a different investigation."
        ))
        console.print(Panel(
            Group(*lines), box=rich_box.ROUNDED, border_style="yellow",
            title="plan change requested", title_align="left", padding=(0, 1),
        ))


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


async def run_investigation(
    *,
    spec_path: str,
    runs_root: Path,
    config: AgentConfig,
    console: Console,
    flags: AblationFlags | None = None,
    resume: bool | None = None,
    quiet: bool = False,
    verbose: bool = False,
    wallclock: int = 0,
    prompt: str = "Begin executing the investigation.",
) -> int:
    """Execute an approved spec to completion and render the results inline.

    Reusable by the CLI (``python -m autointerp_agent.investigation``) and by
    the interactive REPL (which runs it in-session after finalize, then
    returns the user to the prompt). The caller is responsible for having a
    usable model/key — this does not run the setup flow.
    """
    handle, spec, state = prepare_run(
        spec_path, runs_root=runs_root, resume=resume, flags=flags
    )
    console.print(
        f"[bold]Run[/bold]: {handle.run_id}  → {handle.root}  "
        f"[ablation: {handle.flags.label()}]"
    )
    if state.terminal_state is not None:
        console.print(
            f"[yellow]Run already terminal:[/yellow] {state.terminal_state.value}"
        )
        if not quiet:
            _render_inline_results(console, handle, state)
        return 0

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

    cost_path = handle.root / "cost.json"
    cost_tracker = CostTracker.load_or_new(cost_path, run_id=handle.run_id)
    if not quiet and not verbose:
        _print_run_header(console, spec, handle)

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

        # One turn driver per display mode. The transcript writer always runs
        # (disk capture is independent of console rendering).
        _run_turn = _make_turn_runner(
            quiet, verbose, config, context, router, console, handle, cost_tracker
        )

        reason = "error"
        answer = ""
        try:
            drive = _drive_to_completion(handle, _run_turn, prompt)
            if wallclock and wallclock > 0:
                answer, reason = await asyncio.wait_for(drive, timeout=wallclock)
            else:
                answer, reason = await drive
        except asyncio.TimeoutError:
            answer, reason = ("[run killed: wallclock cap reached]", "wallclock")
        except Exception as exc:  # noqa: BLE001 — leave the run resumable
            answer, reason = (f"[turn errored: {format_llm_error(exc)}]", "error")
            console.print(f"[red]{answer}[/red]")
        finally:
            # Always persist the cost snapshot — partial runs are still billable.
            cost_tracker.write(cost_path)

    final_state = read_state(handle.state_path)
    report_path = write_report(handle) if final_state.terminal_state is not None else None

    if not quiet:
        if answer and not answer.startswith("["):
            from .repl import render_answer

            render_answer(console, answer)
        console.print(cost_tracker.format_summary())
        _print_run_summary(
            console, handle, final_state, reason, report_path, spec_arg=spec_path
        )
        _render_inline_results(console, handle, final_state)
    elif report_path is not None:
        console.print(f"[green]Report written:[/green] {report_path}")
    return 0


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

    config = _load_cli_config(args)
    ready = await ensure_model_ready(
        config, console, interactive=sys.stdin.isatty() and not args.quiet
    )
    if ready is None:
        return 2

    return await run_investigation(
        spec_path=args.spec, runs_root=Path(args.runs_root), config=ready,
        console=console, flags=flags, resume=resume, quiet=args.quiet,
        verbose=args.verbose, wallclock=args.wallclock, prompt=args.prompt,
    )


def _make_turn_runner(
    quiet: bool,
    verbose: bool,
    config: AgentConfig,
    context: ContextManager,
    router: ToolRouter,
    console: Console,
    handle: RunHandle,
    cost_tracker: CostTracker,
) -> Callable[[str], Awaitable[str]]:
    """Build the per-turn runner for the active display mode.

    - ``--verbose``: the legacy RunObserver console format (disk + console).
    - ``--quiet``: disk transcript only, no console.
    - default: the live spinner + semantic tool feed, with a separate disk
      transcript writer fanned in.
    """
    if verbose:
        observer = RunObserver(handle, console=console, verbose=True)

        async def run_turn(prompt: str) -> str:
            return await run_agent_turn(
                prompt, config, context, router,
                observer=observer, cost_tracker=cost_tracker,
            )

        return run_turn

    disk_observer = RunObserver(handle, console=None)
    if quiet:

        async def run_turn(prompt: str) -> str:
            return await run_agent_turn(
                prompt, config, context, router,
                observer=disk_observer, cost_tracker=cost_tracker,
            )

        return run_turn

    from .repl import run_live_turn

    async def run_turn(prompt: str) -> str:
        return await run_live_turn(
            prompt, config, context, router, console,
            cost_tracker=cost_tracker, extra_observer=disk_observer,
            initial_label="running the investigation…",
        )

    return run_turn


def _load_cli_config(args: argparse.Namespace) -> AgentConfig:
    load_env_files()  # credentials/.env load even when no config file exists
    config_path = Path(args.config)
    config = (
        load_config(config_path)
        if config_path.exists()
        else AgentConfig(model_name=env_default_model())
    )
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
