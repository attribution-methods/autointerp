"""Tests for the investigation pipeline entry point and Tier-2 tool wrappers."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import init_run
from autointerp.pipelines.investigation.main import (
    build_system_prompt,
    prepare_run,
    render_spec_summary,
)
from autointerp.pipelines.investigation.tools import create_investigation_tools
from autointerp.schemas import BehaviorSpec
from autointerp.spec import (
    AbortPredicate,
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


def _spec_path(tmp_path: Path, *, status: SpecStatus = SpecStatus.APPROVED) -> Path:
    pred = AbortPredicate(
        predicate_id="abort-acc-low",
        description="d",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.5,
    )
    spec = InvestigationSpec(
        spec_id="entry-fixture",
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
                notes="run accuracy on dev",
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
        abort_if=[pred, "advisory string"],
        budget=Budget(max_tool_calls=20),
        status=status,
        approval=(
            Approval(approver="t", approver_kind="agent", approved_at=_now())
            if status == SpecStatus.APPROVED
            else None
        ),
    )
    p = tmp_path / "spec.json"
    p.write_text(spec.model_dump_json(indent=2))
    return p


# ---- main / system prompt --------------------------------------------------


def test_render_spec_summary_includes_key_pieces(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    md = render_spec_summary(spec)
    assert "entry-fixture" in md
    assert "black_box" in md
    assert "accuracy" in md
    assert "Success criteria" in md
    assert "(typed) abort-acc-low" in md
    assert "(advisory) advisory string" in md


def test_build_system_prompt_has_rules_and_spec(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    prompt = build_system_prompt(spec, "run-x")
    assert "Inviolable rules" in prompt
    assert "compute_metric" in prompt
    assert "request_spec_revision" in prompt
    assert "Run: run-x" in prompt
    assert "entry-fixture" in prompt


def test_prepare_run_fresh(tmp_path: Path) -> None:
    spec_path = _spec_path(tmp_path)
    handle, spec, state = prepare_run(spec_path, runs_root=tmp_path / "runs")
    assert handle.run_id == "entry-fixture_rev1"
    assert state.current_stage_idx == 0


def test_prepare_run_resume_default(tmp_path: Path) -> None:
    spec_path = _spec_path(tmp_path)
    runs_root = tmp_path / "runs"
    h1, *_ = prepare_run(spec_path, runs_root=runs_root)
    # Second call with default resume=auto should load, not error.
    h2, _, _ = prepare_run(spec_path, runs_root=runs_root)
    assert h1.root == h2.root


def test_prepare_run_no_resume_refuses_existing(tmp_path: Path) -> None:
    spec_path = _spec_path(tmp_path)
    runs_root = tmp_path / "runs"
    prepare_run(spec_path, runs_root=runs_root)
    with pytest.raises(FileExistsError):
        prepare_run(spec_path, runs_root=runs_root, resume=False)


def test_prepare_run_resume_required_but_missing(tmp_path: Path) -> None:
    spec_path = _spec_path(tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare_run(spec_path, runs_root=tmp_path / "runs", resume=True)


def test_prepare_run_refuses_unapproved(tmp_path: Path) -> None:
    sp = _spec_path(tmp_path, status=SpecStatus.DRAFT)
    with pytest.raises(ValueError, match="not approved"):
        prepare_run(sp, runs_root=tmp_path / "runs")


# ---- Tier-2 tool wrappers --------------------------------------------------


def test_create_investigation_tools_registers_all(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = create_investigation_tools(handle)
    names = {t.name for t in tools}
    assert names == {
        "commit_artifact",
        "compute_metric",
        "compute_and_commit_metric",
        "evaluate_criterion",
        "advance_stage",
        "current_stage",
        "get_state",
        "get_budget",
        "get_progress",
        "request_spec_revision",
    }
    # Each tool exposes a JSON schema and a handler.
    for tool in tools:
        assert tool.handler is not None
        as_openai = tool.as_openai_tool()
        assert as_openai["function"]["name"] == tool.name
        assert "parameters" in as_openai["function"]


def test_tool_compute_then_commit_roundtrip(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}

    out, ok = asyncio.run(
        tools["compute_metric"].handler(
            {
                "metric": "accuracy",
                "metric_id": "a1",
                "inputs": {"predictions": [1, 1, 0, 1], "labels": [1, 0, 0, 1]},
            }
        )
    )
    assert ok is True
    parsed = json.loads(out)
    assert parsed["payload"]["value"] == 0.75
    token = parsed["provenance_token"]

    out2, ok2 = asyncio.run(
        tools["commit_artifact"].handler(
            {
                "kind": "MetricResult",
                "payload": parsed["payload"],
                "split": "dev",
                "provenance_token": token,
            }
        )
    )
    assert ok2 is True
    parsed2 = json.loads(out2)
    assert parsed2["relpath"].endswith("metric_a1.json")


def test_tool_compute_metric_unknown_returns_ok_false(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}
    out, ok = asyncio.run(
        tools["compute_metric"].handler(
            {"metric": "nope", "metric_id": "x", "inputs": {}}
        )
    )
    assert ok is False
    assert "MetricRegistryError" in out


def test_tool_advance_stage_blocks_then_succeeds(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}
    out, ok = asyncio.run(tools["advance_stage"].handler({}))
    assert ok is False
    assert "has not committed" in out

    out, _ = asyncio.run(
        tools["compute_metric"].handler(
            {
                "metric": "accuracy",
                "metric_id": "a",
                "inputs": {"predictions": [1, 1], "labels": [1, 1]},
            }
        )
    )
    parsed = json.loads(out)
    asyncio.run(
        tools["commit_artifact"].handler(
            {
                "kind": "MetricResult",
                "payload": parsed["payload"],
                "split": "dev",
                "provenance_token": parsed["provenance_token"],
            }
        )
    )
    asyncio.run(
        tools["evaluate_criterion"].handler({"criterion_id": "c1"})
    )
    out, ok = asyncio.run(tools["advance_stage"].handler({}))
    assert ok is True
    parsed_adv = json.loads(out)
    assert parsed_adv["terminal_state"] == "completed"


def test_tool_request_revision_terminates(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}
    out, ok = asyncio.run(
        tools["request_spec_revision"].handler({"reason": "mismatch"})
    )
    assert ok is True
    parsed = json.loads(out)
    assert parsed["reason"] == "mismatch"
    state_out, _ = asyncio.run(tools["get_state"].handler({}))
    state_parsed = json.loads(state_out)
    assert state_parsed["terminal_state"] == "revision_requested"


def test_tool_current_stage_view_is_readonly(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}
    out, ok = asyncio.run(tools["current_stage"].handler({}))
    parsed = json.loads(out)
    assert ok is True
    assert parsed["stage"] == "black_box"
    assert parsed["metrics"] == ["accuracy"]
    assert parsed["committed_metric_names"] == []


def test_tool_evaluate_criterion_missing_id(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}
    out, ok = asyncio.run(tools["evaluate_criterion"].handler({}))
    assert ok is False
    assert "criterion_id" in out


def test_tool_commit_artifact_handler_validation(tmp_path: Path) -> None:
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}
    out, ok = asyncio.run(
        tools["commit_artifact"].handler(
            {"kind": "PromptBatch", "payload": {"batch_id": "incomplete"}}
        )
    )
    assert ok is False
    assert "ArtifactGateError" in out


def test_get_budget_shows_unlimited_not_zero(tmp_path: Path) -> None:
    """An unset budget must read as 'unlimited', not as all-zero counters a
    weak agent misreads as 'zero budget' and bails on."""
    spec = InvestigationSpec.model_validate_json(_spec_path(tmp_path).read_text())
    handle = init_run(spec, runs_root=tmp_path / "runs")
    tools = {t.name: t for t in create_investigation_tools(handle)}
    out, ok = asyncio.run(tools["get_budget"].handler({}))
    assert ok
    data = json.loads(out)
    assert "limits" in data and "consumed_so_far" in data
    # The fixture spec sets max_tool_calls=20; the rest are unlimited.
    assert data["limits"]["max_tool_calls"] == 20
    assert data["limits"]["max_gpu_seconds"] == "unlimited"
    assert data["consumed_so_far"]["tool_calls"] == 0
    assert "unlimited" in data["note"]
