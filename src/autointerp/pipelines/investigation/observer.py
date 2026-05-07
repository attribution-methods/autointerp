"""Live + persistent transcript capture for an investigation run.

The investigation gates already write a structured ``log.jsonl`` for every
*successful gated tool call*. This observer is complementary: it captures
the **agent transcript** — every assistant turn (full LLM text + tool-call
summaries) and every tool invocation (full args + full result string) — so a
debugger can replay the run after the fact, including bash commands and
arbitrary tool calls that don't go through the typed gates.

Three sinks:

- ``assistant_turns.jsonl`` — one line per assistant message.
- ``tool_invocations.jsonl`` — one line per tool call (args + truncated
  preview of the output).
- ``tool_invocations/<iteration>_<idx>_<tool>.txt`` — full output body for
  every tool call (untruncated up to the wire limit).

Optionally streams a one-liner per event to a Rich console.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .run_dir import RunHandle
from .state import now_iso


_PREVIEW_CHARS = 400  # full preview length for jsonl artifacts
_CONSOLE_PREVIEW_CHARS = 110  # tighter preview for live console
_ASSISTANT_CONSOLE_CHARS = 180  # assistant reasoning shown on console


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _humanize_chars(n: int) -> str:
    if n < 1000:
        return f"{n}c"
    if n < 1_000_000:
        return f"{n/1000:.1f}kc"
    return f"{n/1_000_000:.1f}Mc"


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:64]


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry, separators=(",", ":"), default=str) + "\n")


class RunObserver:
    """Persistent transcript writer + optional live console stream.

    Attach to ``run_agent_turn(..., observer=...)``. All file writes are
    append-only and confined to the run directory.
    """

    def __init__(
        self,
        handle: RunHandle,
        *,
        console: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self.handle = handle
        self.console = console
        self.verbose = verbose
        self.assistant_path = handle.root / "assistant_turns.jsonl"
        self.invocations_path = handle.root / "tool_invocations.jsonl"
        self.bodies_dir = handle.root / "tool_invocations"
        self.bodies_dir.mkdir(parents=True, exist_ok=True)
        # Track which iteration label has already been printed so we only show
        # the iteration number once per turn (on the first piece of content).
        self._printed_iter_label: int | None = None
        self._current_iter: int = 0
        # Master-agent token + cost accumulators. Sub-agent cost is tracked
        # separately under each discovery session's loop_log.jsonl; this is
        # the master's LiteLLM spend only.
        self.cost_summary_path = handle.root / "cost_summary.json"
        self._n_turns: int = 0
        self._cost_usd_total: float = 0.0
        self._tokens_in_total: int = 0
        self._tokens_out_total: int = 0
        self._tokens_cache_read_total: int = 0
        self._tokens_cache_create_total: int = 0

    # -- console helpers ----------------------------------------------------

    def _print(self, message: str, *, style: str | None = None) -> None:
        if self.console is None:
            return
        try:
            if style:
                self.console.print(message, style=style)
            else:
                self.console.print(message)
        except Exception:
            pass

    # -- hook methods (called by run_agent_turn via _emit) ------------------

    def _iter_prefix(self, iteration: int) -> str:
        """Print iteration label once per turn; subsequent lines get spacer."""
        if self._printed_iter_label == iteration:
            return "    "
        self._printed_iter_label = iteration
        return f"[dim]{iteration:>3}[/dim] "

    def on_iteration_start(self, iteration: int) -> None:
        self._current_iter = iteration
        if self.verbose:
            self._print(f"[dim]── iteration {iteration} ──[/dim]")

    def on_assistant(self, iteration: int, message: dict[str, Any]) -> None:
        ts = now_iso()
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        tc_summary = [
            {
                "id": tc.get("id"),
                "name": (tc.get("function") or {}).get("name"),
                "args_raw": (tc.get("function") or {}).get("arguments"),
            }
            for tc in tool_calls
        ]
        usage = message.get("_usage") or {}
        cost_usd = message.get("_cost_usd")

        record: dict[str, Any] = {
            "ts": ts,
            "iteration": iteration,
            "content": content,
            "tool_calls": tc_summary,
        }
        if usage:
            record["usage"] = usage
        if cost_usd is not None:
            record["cost_usd"] = float(cost_usd)
        _append_jsonl(self.assistant_path, record)

        # Update running totals + write cost_summary.json after every turn
        # so a partial run still has cost data on disk if the master loop
        # is killed mid-iteration.
        self._n_turns += 1
        if cost_usd is not None:
            self._cost_usd_total += float(cost_usd)
        if usage:
            self._tokens_in_total += int(usage.get("prompt_tokens") or 0)
            self._tokens_out_total += int(usage.get("completion_tokens") or 0)
            self._tokens_cache_read_total += int(
                usage.get("cache_read_input_tokens") or 0
            )
            self._tokens_cache_create_total += int(
                usage.get("cache_creation_input_tokens") or 0
            )
        self._write_cost_summary()
        if self.verbose:
            if content:
                self._print(
                    f"[bold cyan]assistant[/bold cyan]: {_preview(str(content))}"
                )
            for tc in tc_summary:
                name = tc.get("name") or "?"
                self._print(
                    f"  [yellow]→ {name}[/yellow]  (id={tc.get('id')})",
                    style=None,
                )
            return
        # Compact mode: only print assistant reasoning. Tool calls are printed
        # by on_tool_call along with their result on a single line.
        text = str(content).strip() if content else ""
        if text:
            prefix = self._iter_prefix(iteration)
            self._print(
                f"{prefix}[italic]{_preview(text, _ASSISTANT_CONSOLE_CHARS)}[/italic]"
            )

    def on_tool_call(
        self,
        iteration: int,
        call_idx: int,
        call_id: str,
        name: str,
        args: dict[str, Any],
        output: str,
        ok: bool,
    ) -> None:
        ts = now_iso()
        body_filename = (
            f"{iteration:04d}_{call_idx:02d}_{_safe_name(str(name))}_{_safe_name(str(call_id))}.txt"
        )
        body_path = self.bodies_dir / body_filename
        try:
            body_path.write_text(output)
        except Exception:
            # Never break the loop on disk errors — record what we can in jsonl.
            body_path = None  # type: ignore[assignment]

        _append_jsonl(
            self.invocations_path,
            {
                "ts": ts,
                "iteration": iteration,
                "call_idx": call_idx,
                "call_id": call_id,
                "tool": name,
                "args": args,
                "ok": ok,
                "output_chars": len(output),
                "output_preview": _preview(output),
                "body_ref": (
                    body_path.relative_to(self.handle.root).as_posix()
                    if body_path is not None
                    else None
                ),
            },
        )
        if self.verbose:
            verdict = "[green]ok[/green]" if ok else "[red]err[/red]"
            self._print(
                f"  [yellow]← {name}[/yellow] {verdict} "
                f"({len(output)} chars) {_preview(output)}"
            )
            return
        # Compact mode: tool name + status + size + output preview, all one line.
        # Iteration prefix only on the first line of this turn (handled in
        # _iter_prefix). Tool name padded so columns line up.
        prefix = self._iter_prefix(iteration)
        verdict_label = "ok " if ok else "err"
        verdict_style = "green" if ok else "red"
        size = _humanize_chars(len(output))
        preview_text = _preview(output, _CONSOLE_PREVIEW_CHARS) or "(no output)"
        preview_style = "dim" if ok else "red"
        self._print(
            f"{prefix}[cyan]{name:<16}[/cyan] [{verdict_style}]{verdict_label}[/{verdict_style}] "
            f"[dim]{size:>6}[/dim]  [{preview_style}]{preview_text}[/{preview_style}]"
        )

    def _write_cost_summary(self) -> None:
        """Atomic-write cost_summary.json. Called after every assistant turn
        so a killed run still leaves a usable cost record on disk."""
        import json
        import tempfile
        payload = {
            "ts": now_iso(),
            "n_turns": self._n_turns,
            "cost_usd": round(self._cost_usd_total, 6),
            "usage": {
                "prompt_tokens": self._tokens_in_total,
                "completion_tokens": self._tokens_out_total,
                "cache_read_input_tokens": self._tokens_cache_read_total,
                "cache_creation_input_tokens": self._tokens_cache_create_total,
            },
            "scope": "master_agent_litellm",
            "note": (
                "Sub-agent (discover_features) costs are NOT included here; "
                "see runs/<id>/discovery/<session>/loop_log.jsonl for those."
            ),
        }
        try:
            fd, tmp = tempfile.mkstemp(
                dir=self.cost_summary_path.parent, suffix=".tmp"
            )
            try:
                import os as _os
                with _os.fdopen(fd, "w") as fh:
                    json.dump(payload, fh, indent=2)
                _os.replace(tmp, self.cost_summary_path)
            finally:
                if Path(tmp).exists():
                    Path(tmp).unlink()
        except Exception:
            # Cost recording must never break the run.
            pass

    def on_final(self, iteration: int, final_text: str) -> None:
        _append_jsonl(
            self.assistant_path,
            {
                "ts": now_iso(),
                "iteration": iteration,
                "final": True,
                "content": final_text,
            },
        )
        self._write_cost_summary()
        if self.verbose:
            self._print(f"[bold green]final[/bold green]: {_preview(final_text)}")
            return
        # Compact: a clear separator + the final text. No iteration prefix
        # because the run is ending, not continuing.
        text = _preview(final_text, _ASSISTANT_CONSOLE_CHARS * 2)
        self._print("")
        self._print(f"[bold green]✓ final[/bold green]  {text}")


__all__ = ["RunObserver"]
