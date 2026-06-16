"""Tests for the bash watchdog (stall detection + heartbeat log)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from autointerp_agent.skills import SkillRegistry
from autointerp_agent.tools import ToolRouter, _run_bash_watched


def test_run_bash_completes(tmp_path: Path) -> None:
    out, ok = _run_bash_watched(
        "printf 'hello world\\n'",
        work_dir=str(tmp_path),
        timeout=10,
        stall_seconds=10,
        scratch_dir=None,
    )
    assert ok is True
    assert "hello world" in out


def test_run_bash_surfaces_nonzero_exit(tmp_path: Path) -> None:
    out, ok = _run_bash_watched(
        "false",
        work_dir=str(tmp_path),
        timeout=5,
        stall_seconds=5,
        scratch_dir=None,
    )
    assert ok is False
    assert "[exit 1]" in out


def test_run_bash_hard_timeout(tmp_path: Path) -> None:
    out, ok = _run_bash_watched(
        "sleep 5",
        work_dir=str(tmp_path),
        timeout=1,
        stall_seconds=10,
        scratch_dir=None,
    )
    assert ok is False
    assert "timed out" in out.lower()


def test_run_bash_stall_detection(tmp_path: Path) -> None:
    # No output for >stall_seconds → killed as stalled before hitting hard timeout.
    out, ok = _run_bash_watched(
        "sleep 3",
        work_dir=str(tmp_path),
        timeout=10,
        stall_seconds=1,
        scratch_dir=None,
    )
    assert ok is False
    assert "stall" in out.lower()


def test_run_bash_heartbeat_streams_to_disk(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    out, ok = _run_bash_watched(
        "for i in 1 2 3; do printf 'line%s\\n' $i; sleep 0.05; done",
        work_dir=str(tmp_path),
        timeout=10,
        stall_seconds=10,
        scratch_dir=scratch,
    )
    assert ok is True
    logs = list(scratch.glob("bash_*.log"))
    assert len(logs) == 1
    body = logs[0].read_text()
    assert "line1" in body and "line3" in body


def test_router_threads_scratch_dir(tmp_path: Path) -> None:
    """Calling bash via the router uses router.scratch_dir for heartbeat."""
    r = ToolRouter(skill_registry=SkillRegistry(skills={}), auto_approve=True)
    r.scratch_dir = tmp_path / "scratch"
    out, ok = asyncio.run(
        r.call_tool("bash", {"command": "printf hello", "work_dir": str(tmp_path)})
    )
    assert ok is True
    assert "hello" in out
    assert (tmp_path / "scratch").exists()
    assert any((tmp_path / "scratch").glob("bash_*.log"))


def test_bash_handles_empty_and_bad_work_dir() -> None:
    import asyncio

    from autointerp_agent.tools import _bash_handler

    handler = _bash_handler(None)
    # Empty work_dir (the FileNotFoundError('') trigger) normalizes to "."
    out, ok = asyncio.run(handler({"command": "echo hi", "work_dir": ""}))
    assert ok and "hi" in out
    # A non-existent directory is reported, not crashed on.
    out, ok = asyncio.run(handler({"command": "echo hi", "work_dir": "/no/such/dir"}))
    assert not ok and "not a directory" in out


def test_bash_rejects_mangled_heredoc() -> None:
    """The exact field failure: a heredoc with literal '\\n' instead of real
    newlines. Caught with an actionable message, not run as broken bash."""
    import asyncio

    from autointerp_agent.tools import _bash_handler

    handler = _bash_handler(None)
    bad = 'python - <<"PY"\\nimport os\\nprint(1)\\nPY'  # literal backslash-n
    out, ok = asyncio.run(handler({"command": bad}))
    assert ok is False
    assert "literal" in out.lower() and "write_file" in out
    assert "spec revision" in out.lower()  # tells it this is NOT a spec problem

    # A real heredoc (actual newlines) is not tripped by the guard.
    out2, ok2 = asyncio.run(handler({"command": "cat <<EOF\nhi\nEOF"}))
    assert ok2 is True and "hi" in out2


def test_bash_runs_inside_run_dir_not_repo(tmp_path) -> None:
    """During an investigation, the agent's relative bash writes must land in
    the run dir (runs/<id>/), not pollute the repo's own scripts/ dir."""
    import asyncio
    import types

    from autointerp_agent.tools import _bash_handler

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    router = types.SimpleNamespace(run_dir=run_dir, scratch_dir=run_dir / "scratch")
    handler = _bash_handler(router)
    out, ok = asyncio.run(handler(
        {"command": "mkdir -p scripts && echo hi > scripts/x.txt && pwd"}
    ))
    assert ok, out
    assert (run_dir / "scripts" / "x.txt").read_text().strip() == "hi"
    assert str(run_dir.resolve()) in out  # pwd is the run dir
