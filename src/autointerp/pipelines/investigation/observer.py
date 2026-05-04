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


_PREVIEW_CHARS = 400


def _preview(text: str) -> str:
    text = text.replace("\n", " ")
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[: _PREVIEW_CHARS - 3] + "..."


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

    def __init__(self, handle: RunHandle, *, console: Any | None = None) -> None:
        self.handle = handle
        self.console = console
        self.assistant_path = handle.root / "assistant_turns.jsonl"
        self.invocations_path = handle.root / "tool_invocations.jsonl"
        self.bodies_dir = handle.root / "tool_invocations"
        self.bodies_dir.mkdir(parents=True, exist_ok=True)

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

    def on_iteration_start(self, iteration: int) -> None:
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
        _append_jsonl(
            self.assistant_path,
            {
                "ts": ts,
                "iteration": iteration,
                "content": content,
                "tool_calls": tc_summary,
            },
        )
        if content:
            self._print(f"[bold cyan]assistant[/bold cyan]: {_preview(str(content))}")
        for tc in tc_summary:
            name = tc.get("name") or "?"
            self._print(f"  [yellow]→ {name}[/yellow]  (id={tc.get('id')})", style=None)

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
        verdict = "[green]ok[/green]" if ok else "[red]err[/red]"
        self._print(
            f"  [yellow]← {name}[/yellow] {verdict} "
            f"({len(output)} chars) {_preview(output)}"
        )

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
        self._print(f"[bold green]final[/bold green]: {_preview(final_text)}")


__all__ = ["RunObserver"]
