"""Ablation feature-flags: all-on is byte-identical; each flag removes
exactly its gate + announced rule. See docs/scaffold_faithfulness_eval.md.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import (
    AblationFlags,
    ArtifactGateError,
    CriterionGateError,
    commit_artifact,
    compute_metric,
    evaluate_criterion,
    init_run,
    load_run,
)
from autointerp.pipelines.investigation.main import (
    SYSTEM_PROMPT_HEADER,
    build_header,
    build_system_prompt,
)
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


def _spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="ablation-fixture",
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
            StageSpec(
                stage=InvestigationStage.VALIDATION,
                pattern=PatternId.BLACKBOX_THEN_PATCHING,
                tools=[ToolName.ACTIVATION_PATCHING],
                metrics=[MetricName.FAITHFULNESS],
            ),
        ],
        success_criteria=[
            Criterion(
                criterion_id="circuit-faithfulness",
                description="d",
                metric=MetricName.FAITHFULNESS,
                comparator=">=",
                threshold=0.7,
                on_split="heldout",
            ),
        ],
        abort_if=[],
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(
            approver="t",
            approver_kind="agent",
            approved_at=datetime.now(timezone.utc),
        ),
    )


def _faithfulness_payload(handle, ratio: float):
    return compute_metric(
        handle,
        metric=MetricName.FAITHFULNESS,
        metric_id="f-1",
        inputs={
            "circuit_metric": ratio,
            "full_model_metric": 1.0,
            "corrupted_metric": 0.0,
        },
    )


# --- the load-bearing invariant: all-on == original scaffold ---------------


def test_all_on_header_is_byte_identical() -> None:
    assert build_header() == SYSTEM_PROMPT_HEADER
    assert build_header(AblationFlags()) == SYSTEM_PROMPT_HEADER
    assert AblationFlags().all_on is True
    # build_system_prompt with no flags is unchanged (existing contract).
    prompt = build_system_prompt(_spec(), "run-x")
    assert "Inviolable rules" in prompt and "Run: run-x" in prompt


def test_parse_and_label() -> None:
    assert AblationFlags.parse("full").label() == "full"
    assert AblationFlags.parse(None).all_on is True
    a = AblationFlags.parse("A")
    assert a.freeze_spec is False and a.provenance_metrics is True
    assert a.label() == "loo-A"
    assert AblationFlags.parse("A,C").label() == "loo-AC"
    assert AblationFlags.leave_out("B").provenance_metrics is False
    with pytest.raises(ValueError):
        AblationFlags.parse("Z")


# --- header surgery: drop the rule, renumber, drop the announcement --------


def test_loo_B_header_drops_provenance_rule_and_renumbers() -> None:
    h = build_header(AblationFlags.leave_out("B"))
    assert "one-time provenance" not in h
    assert "evaluate_criterion` runs once" in h  # always-on rule survives
    # rules renumbered contiguously — no "1." then jump.
    rules_block = h.split("# Inviolable rules", 1)[1].split("# How to work", 1)[0]
    nums = [ln.split(".")[0].strip() for ln in rules_block.strip().splitlines()
            if ln.strip()[:2].rstrip(".").isdigit() and ln.strip()[1] in ". "]
    leading = [s for s in nums if s.isdigit()]
    assert leading == [str(i) for i in range(1, len(leading) + 1)]


def test_loo_A_header_drops_spec_rules_and_revision_prose() -> None:
    h = build_header(AblationFlags.leave_out("A"))
    assert "The spec is read-only" not in h
    assert "are immutable for this run" not in h
    assert "call `request_spec_revision` with a clear" not in h
    assert "frozen approved spec (read-only)" not in h
    assert "writable in this ablation run" in h
    # untouched mechanisms still announced
    assert "one-time provenance" in h
    assert "Cross-split contamination is mechanically blocked" in h


# --- the three seams behave differently only when their flag is off --------


def test_loo_A_spec_file_is_writable(tmp_path: Path) -> None:
    h_full = init_run(_spec(), runs_root=tmp_path / "full")
    assert not os.access(h_full.spec_path, os.W_OK)  # frozen

    h_a = init_run(_spec(), runs_root=tmp_path / "a", flags=AblationFlags.leave_out("A"))
    assert os.access(h_a.spec_path, os.W_OK)  # editable under flag A off
    # persisted + restored on resume, and not re-frozen
    h_a2, _spec2, st = load_run(h_a.root)
    assert st.ablation_flags.freeze_spec is False
    assert h_a2.flags.freeze_spec is False
    assert os.access(h_a.spec_path, os.W_OK)


def test_loo_B_metric_commits_without_token(tmp_path: Path) -> None:
    # all-on: a MetricResult with no provenance token is refused.
    h_full = init_run(_spec(), runs_root=tmp_path / "full")
    payload, _tok = _faithfulness_payload(h_full, 0.9)
    with pytest.raises(ArtifactGateError, match="provenance_token"):
        commit_artifact(h_full, "MetricResult", payload, split="heldout")

    # flag B off: the same self-reported value commits fine (number fab).
    h_b = init_run(_spec(), runs_root=tmp_path / "b", flags=AblationFlags.leave_out("B"))
    payload_b, _ = _faithfulness_payload(h_b, 0.9)
    ref = commit_artifact(h_b, "MetricResult", payload_b, split="heldout")
    assert ref.kind == "MetricResult"


def test_loo_C_allows_cross_split_criterion(tmp_path: Path) -> None:
    # all-on: a heldout criterion cannot be satisfied by a dev-tagged metric.
    h_full = init_run(_spec(), runs_root=tmp_path / "full")
    p, t = _faithfulness_payload(h_full, 0.9)
    commit_artifact(h_full, "MetricResult", p, split="dev", provenance_token=t)
    with pytest.raises(CriterionGateError):
        evaluate_criterion(h_full, "circuit-faithfulness")

    # flag C off: the dev metric now satisfies the heldout criterion.
    h_c = init_run(_spec(), runs_root=tmp_path / "c", flags=AblationFlags.leave_out("C"))
    p2, t2 = _faithfulness_payload(h_c, 0.9)
    commit_artifact(h_c, "MetricResult", p2, split="dev", provenance_token=t2)
    rec = evaluate_criterion(h_c, "circuit-faithfulness")
    assert rec.value == pytest.approx(0.9)
