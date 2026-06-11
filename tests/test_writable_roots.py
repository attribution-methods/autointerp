"""Tests for the range-restricted write_file / edit_file gate."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autointerp.utils.paths import is_within, normalize_safe_path
from autointerp_agent.skills import SkillRegistry
from autointerp_agent.tools import ToolRouter


@pytest.fixture
def router(tmp_path: Path) -> ToolRouter:
    r = ToolRouter(skill_registry=SkillRegistry(skills={}), auto_approve=True)
    r.writable_roots = [tmp_path / "scripts", tmp_path / "scratch"]
    return r


def test_normalize_rejects_relative_paths() -> None:
    assert normalize_safe_path("relative/foo") is None
    assert normalize_safe_path("") is None


def test_normalize_rejects_null_bytes() -> None:
    assert normalize_safe_path("/tmp/foo\0bar") is None


def test_normalize_rejects_unc_and_drive_root() -> None:
    assert normalize_safe_path("\\\\server\\share") is None
    assert normalize_safe_path("//server/share") is None
    assert normalize_safe_path("C:\\") is None


def test_normalize_resolves_dotdot(tmp_path: Path) -> None:
    target = (tmp_path / "a" / ".." / "b").as_posix()
    safe = normalize_safe_path(target)
    assert safe is not None
    assert ".." not in str(safe)


def test_is_within_directory(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    assert is_within(root / "deep" / "x.py", [root])


def test_is_within_rejects_sibling(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert not is_within(b / "x", [a])


def test_is_within_file_root_exact_match(tmp_path: Path) -> None:
    log = tmp_path / "INVESTIGATION_LOG.md"
    log.write_text("hi")
    assert is_within(log, [log])
    assert not is_within(tmp_path / "other.md", [log])


def test_writable_root_blocks_outside_path(router: ToolRouter, tmp_path: Path) -> None:
    out = tmp_path / "outside" / "x.py"
    out.parent.mkdir(parents=True)
    msg, ok = asyncio.run(
        router.call_tool("write_file", {"path": str(out), "content": "hi"})
    )
    assert ok is False
    assert "range-restricted" in msg


def test_writable_root_allows_scripts(router: ToolRouter, tmp_path: Path) -> None:
    target = tmp_path / "scripts" / "run.py"
    target.parent.mkdir(parents=True)
    msg, ok = asyncio.run(
        router.call_tool(
            "write_file", {"path": str(target), "content": "print('hi')\n"}
        )
    )
    assert ok is True, msg
    assert target.read_text() == "print('hi')\n"


def test_writable_root_blocks_unsafe_path(router: ToolRouter) -> None:
    msg, ok = asyncio.run(
        router.call_tool("write_file", {"path": "relative.py", "content": "hi"})
    )
    assert ok is False
    assert "unsafe" in msg.lower()


def test_no_writable_roots_means_unrestricted(tmp_path: Path) -> None:
    r = ToolRouter(skill_registry=SkillRegistry(skills={}), auto_approve=True)
    # No router.writable_roots set: REPL / Stage 0 behavior. Anywhere allowed.
    target = tmp_path / "anywhere.py"
    msg, ok = asyncio.run(
        r.call_tool("write_file", {"path": str(target), "content": "hi"})
    )
    assert ok is True, msg
    assert target.read_text() == "hi"


def test_tool_aliases_resolved(tmp_path: Path) -> None:
    from autointerp_agent.tools import ToolSpec

    r = ToolRouter(skill_registry=SkillRegistry(skills={}), auto_approve=True)

    async def _h(args: dict) -> tuple[str, bool]:
        return f"called: {args}", True

    r.register_tool(
        ToolSpec(
            name="commit_artifact",
            description="commit",
            parameters={"type": "object", "properties": {}},
            aliases=["commit_finding"],
            handler=_h,
        )
    )
    out, ok = asyncio.run(r.call_tool("commit_finding", {"x": 1}))
    assert ok and "called" in out
