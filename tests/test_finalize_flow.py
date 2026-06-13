"""Regression tests for the Stage 0 approval flow — guarding against the
"approve the same plan twice" bug.

Two root causes are covered:

1. ``validate_spec`` used to run only a structural build, while
   ``finalize_spec`` *additionally* ran metric-producibility guards. A plan
   could validate "OK", get presented and approved, then fail at finalize —
   forcing a silent rebuild and a second approval. Now both run the same
   guards, so a plan that validates is a plan that finalizes.

2. ``finalize_spec`` used to be two-phase (render, then ``user_confirmed=true``
   to write), which a weak planner narrated to the user as two separate
   approvals. It is single-phase now: one call validates and writes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from autointerp.schemas import BehaviorSpec
from autointerp.spec import (
    Budget,
    ContrastSpec,
    Criterion,
    DatasetSpec,
    InvestigationSpec,
    InvestigationStage,
    MetricName,
    ModelRef,
    PatternId,
    StageSpec,
    ToolName,
)
from autointerp_agent import stage0_tools as s0


def _draft_data(stage: InvestigationStage, pattern: PatternId,
                tools: list[ToolName], metrics: list[MetricName]) -> dict:
    """A structurally complete InvestigationSpec as a draft dict (no status /
    approval — those are set at finalize)."""
    spec = InvestigationSpec(
        spec_id="surprise-flow",
        revision=1,
        question="How does the model represent surprise in text?",
        hypothesis="Surprise shifts mid-layer activations.",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="gpt2"),
        dataset=DatasetSpec(dataset_id="ds", source="generated", n_samples=10),
        contrast=ContrastSpec(
            contrast_id="c", positive_template="{x}", negative_template="{y}"
        ),
        stages=[StageSpec(stage=stage, pattern=pattern, tools=tools, metrics=metrics)],
        success_criteria=[Criterion(
            criterion_id="c1", description="d", metric=MetricName.ACCURACY,
            comparator=">=", threshold=0.5, on_split="dev",
        )],
        budget=Budget(max_tool_calls=50),
    )
    data = json.loads(spec.model_dump_json())
    data.pop("status", None)
    data.pop("approval", None)
    return data


def _load(data: dict) -> None:
    s0.reset_stage0_state()
    s0.current_partial().data.update(data)


# ---------------------------------------------------------------------------
# Bug 1: validate_spec and finalize_spec agree
# ---------------------------------------------------------------------------


def test_validate_spec_flags_unproducible_metric(monkeypatch, tmp_path: Path) -> None:
    """A causal metric in a setup stage (the real doomed-plan shape) must be
    caught by validate_spec, not only at finalize."""
    monkeypatch.chdir(tmp_path)
    # Causal metric in a SETUP stage — cannot be produced there.
    _load(_draft_data(
        InvestigationStage.SETUP, PatternId.CUSTOM,
        [ToolName.ACTIVATION_CACHE], [MetricName.PATCH_EFFECT_RECOVERY],
    ))
    try:
        msg, ok = asyncio.run(s0._validate_spec({}))
        assert ok  # the tool ran fine; the *spec* is what's not ready
        assert "patch_effect_recovery" in msg
        assert "fix" in msg.lower()  # framed as a must-fix, not "OK"
        assert "OK" not in msg.split("\n")[0]

        # finalize_spec rejects the SAME issue (consistency) — and writes nothing.
        fmsg, fok = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert fok is False
        assert "patch_effect_recovery" in fmsg
        # A rejected finalize writes nothing.
        spec_dir = tmp_path / "outputs" / "specs"
        assert not list(spec_dir.glob("surprise-flow_rev*.json")) if spec_dir.exists() else True
    finally:
        s0.reset_stage0_state()


def test_validate_spec_passes_a_finalizable_plan(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _load(_draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    ))
    try:
        msg, ok = asyncio.run(s0._validate_spec({}))
        assert ok and "ready to finalize" in msg.lower()
    finally:
        s0.reset_stage0_state()


# ---------------------------------------------------------------------------
# Bug 2: finalize_spec is single-phase
# ---------------------------------------------------------------------------


def test_finalize_spec_writes_in_one_call(monkeypatch, tmp_path: Path) -> None:
    """One finalize call on an approved-ready plan writes the spec — no
    'show the rendered spec and ask for approval' second round."""
    monkeypatch.chdir(tmp_path)
    _load(_draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    ))
    try:
        # No user_confirmed flag at all — the call itself is the commit.
        msg, ok = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert ok is True
        assert "AUTOMATICALLY" in msg  # the run launches on its own
        # It is NOT the old phase-1 "ask the user again" message.
        assert "ask for approval" not in msg.lower()
        # The handoff tells the agent to confirm in one line and NOT to end
        # with a question (the run has started and won't wait for a reply).
        assert "do not ask" in msg.lower()
        # The approved spec landed on disk with status APPROVED.
        written = list((tmp_path / "outputs" / "specs").glob("surprise-flow_rev*.json"))
        assert len(written) == 1
        saved = json.loads(written[0].read_text())
        assert saved["status"] == "approved"
        assert saved["approval"]["approver"] == "u@x"
    finally:
        s0.reset_stage0_state()


def test_finalize_spec_requires_approver(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _load(_draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    ))
    try:
        msg, ok = asyncio.run(s0._finalize_spec({"approver": ""}))
        assert ok is False and "approver is required" in msg
    finally:
        s0.reset_stage0_state()
