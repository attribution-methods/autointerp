"""Tests for transcript capture (observer) and agent-loop hook plumbing."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from autointerp.pipelines.investigation import init_run
from autointerp.pipelines.investigation.observer import RunObserver
from autointerp.schemas import BehaviorSpec
from autointerp.spec import (
    Approval,
    Budget,
    ContrastSpec,
    Criterion,
    DatasetSpec,
    InvestigationSpec,
    InvestigationStage,
    MetricName,
    ModelRef,
    PatternId,
    SpecStatus,
    StageSpec,
    ToolName,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="observer-fixture",
        revision=1,
        question="q",
        hypothesis="h",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="m"),
        dataset=DatasetSpec(
            dataset_id="ds", source="generated", n_samples=10, split="dev", seed=1
        ),
        contrast=ContrastSpec(
            contrast_id="c",
            positive_template="{x}",
            negative_template="{y}",
            pairing="matched",
        ),
        stages=[
            StageSpec(
                stage=InvestigationStage.BLACK_BOX,
                pattern=PatternId.BLACKBOX_THEN_PATCHING,
                tools=[ToolName.BLACKBOX_PROBE],
                metrics=[MetricName.ACCURACY],
            ),
        ],
        success_criteria=[
            Criterion(
                criterion_id="c1",
                description="d",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.5,
                on_split="dev",
            )
        ],
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=_now()),
    )


def test_observer_writes_assistant_and_tool_files(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    obs = RunObserver(handle)

    asyncio.run(_drive_observer(obs))

    assistant = (handle.root / "assistant_turns.jsonl").read_text().splitlines()
    invocations = (handle.root / "tool_invocations.jsonl").read_text().splitlines()

    assert len(assistant) == 2  # one in-loop turn + one final
    first = json.loads(assistant[0])
    assert first["iteration"] == 0
    assert first["content"] == "thinking..."
    assert first["tool_calls"][0]["name"] == "compute_metric"

    final = json.loads(assistant[-1])
    assert final["final"] is True
    assert final["content"].startswith("done")

    assert len(invocations) == 1
    inv = json.loads(invocations[0])
    assert inv["tool"] == "compute_metric"
    assert inv["ok"] is True
    assert inv["output_chars"] == len("the full output body")
    assert inv["output_preview"] == "the full output body"
    body = handle.root / inv["body_ref"]
    assert body.read_text() == "the full output body"


def test_observer_handles_long_output_with_preview(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    obs = RunObserver(handle)
    big = "x" * 5000
    obs.on_iteration_start(0)
    obs.on_assistant(0, {"content": "hi", "tool_calls": []})
    obs.on_tool_call(0, 0, "id1", "bash", {"command": "echo x"}, big, True)
    inv = json.loads((handle.root / "tool_invocations.jsonl").read_text().splitlines()[0])
    assert inv["output_chars"] == 5000
    assert inv["output_preview"].endswith("...")
    body_path = handle.root / inv["body_ref"]
    # Full body, not truncated.
    assert len(body_path.read_text()) == 5000


def test_observer_safe_against_console_failures(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)

    class BrokenConsole:
        def print(self, *_args, **_kwargs):
            raise RuntimeError("broken console")

    obs = RunObserver(handle, console=BrokenConsole())
    # Must not raise.
    obs.on_iteration_start(0)
    obs.on_assistant(0, {"content": "hi"})
    obs.on_tool_call(0, 0, "id", "bash", {"command": "echo"}, "out", True)
    obs.on_final(0, "bye")
    # Files still written despite console failures.
    assert (handle.root / "assistant_turns.jsonl").exists()
    assert (handle.root / "tool_invocations.jsonl").exists()


async def _drive_observer(obs: RunObserver) -> None:
    obs.on_iteration_start(0)
    obs.on_assistant(
        0,
        {
            "content": "thinking...",
            "tool_calls": [
                {"id": "call_a", "function": {"name": "compute_metric", "arguments": "{}"}}
            ],
        },
    )
    obs.on_tool_call(
        0, 0, "call_a", "compute_metric", {"metric": "accuracy"}, "the full output body", True
    )
    obs.on_final(1, "done — wrote findings")


def test_run_agent_turn_emits_to_observer(tmp_path: Path) -> None:
    """Drive the agent_loop with a fake LLM and verify all hooks fire."""
    from autointerp_agent.agent_loop import run_agent_turn
    from autointerp_agent.config import AgentConfig
    from autointerp_agent.context import ContextManager
    from autointerp_agent.skills import SkillRegistry
    from autointerp_agent.tools import ToolRouter, ToolSpec
    import autointerp_agent.agent_loop as al

    handle = init_run(_spec(), runs_root=tmp_path)

    events: list[tuple] = []

    class _Recorder:
        def on_iteration_start(self, i): events.append(("start", i))
        def on_assistant(self, i, m): events.append(("assistant", i, m.get("content"), len(m.get("tool_calls") or [])))
        def on_tool_call(self, i, idx, cid, n, a, o, ok):
            events.append(("tool", i, idx, n, ok, o))
        def on_final(self, i, t): events.append(("final", i, t))

    # Fake litellm: first turn calls a tool; second turn returns final text.
    state = {"calls": 0}

    class _FakeMessage:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls
            self.role = "assistant"

        def model_dump(self, exclude_none=False):
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": self.tool_calls,
            }

    class _FakeChoice:
        def __init__(self, msg): self.message = msg

    class _FakeResponse:
        def __init__(self, msg): self.choices = [_FakeChoice(msg)]

    async def fake_acompletion(**kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return _FakeResponse(
                _FakeMessage(
                    content="planning",
                    tool_calls=[
                        {
                            "id": "tc1",
                            "function": {"name": "echo_tool", "arguments": "{\"x\": 1}"},
                        }
                    ],
                )
            )
        return _FakeResponse(_FakeMessage(content="all done"))

    async def echo_handler(args):
        return f"echoed: {args}", True

    # Patch in fake LLM.
    al.acompletion = fake_acompletion  # type: ignore[attr-defined]

    config = AgentConfig(model_name="fake/model", max_iterations=4)
    registry = SkillRegistry(skills={})
    context = ContextManager(skill_registry=registry, default_skill_names=[], system_prompt="sys")

    async def _go():
        async with ToolRouter(skill_registry=registry, auto_approve=True) as router:
            router.register_tool(
                ToolSpec(
                    name="echo_tool",
                    description="echo",
                    parameters={"type": "object", "properties": {}},
                    handler=echo_handler,
                )
            )
            return await run_agent_turn(
                "go", config, context, router, observer=_Recorder()
            )

    final = asyncio.run(_go())
    assert final == "all done"

    kinds = [e[0] for e in events]
    assert kinds.count("start") == 2
    assert kinds.count("assistant") == 2
    assert kinds.count("tool") == 1
    assert kinds.count("final") == 1
    tool_event = next(e for e in events if e[0] == "tool")
    # (tag, iteration, idx, name, ok, output)
    assert tool_event[3] == "echo_tool"
    assert tool_event[4] is True
    assert "echoed" in tool_event[5]


def test_run_agent_turn_observer_failure_does_not_break_loop(tmp_path: Path) -> None:
    """A buggy observer must not crash the agent loop."""
    from autointerp_agent.agent_loop import run_agent_turn
    from autointerp_agent.config import AgentConfig
    from autointerp_agent.context import ContextManager
    from autointerp_agent.skills import SkillRegistry
    from autointerp_agent.tools import ToolRouter
    import autointerp_agent.agent_loop as al

    class _Bad:
        def on_iteration_start(self, i):
            raise RuntimeError("boom-start")
        def on_assistant(self, *a):
            raise RuntimeError("boom-assistant")
        def on_final(self, *a):
            raise RuntimeError("boom-final")

    class _FakeMessage:
        def __init__(self, c): self.content = c
        def model_dump(self, exclude_none=False):
            return {"role": "assistant", "content": self.content, "tool_calls": None}

    class _FakeChoice:
        def __init__(self, m): self.message = m

    class _FakeResponse:
        def __init__(self, m): self.choices = [_FakeChoice(m)]

    async def fake_acompletion(**_):
        return _FakeResponse(_FakeMessage("all good"))

    al.acompletion = fake_acompletion  # type: ignore[attr-defined]

    config = AgentConfig(model_name="fake/model", max_iterations=2)
    registry = SkillRegistry(skills={})
    context = ContextManager(skill_registry=registry, default_skill_names=[], system_prompt="sys")

    async def _go():
        async with ToolRouter(skill_registry=registry, auto_approve=True) as router:
            return await run_agent_turn(
                "go", config, context, router, observer=_Bad()
            )

    assert asyncio.run(_go()) == "all good"
