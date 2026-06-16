"""Tests for the typed AbortPredicate (Stage 0 schema extension)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

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
    typed_abort_predicates,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _spec(*, abort_if: list) -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="abort-fixture",
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
        abort_if=abort_if,
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=_now()),
    )


def test_abort_predicate_typed_round_trip() -> None:
    pred = AbortPredicate(
        predicate_id="abort-acc-low",
        description="Abort if accuracy < 0.75",
        metric=MetricName.ACCURACY,
        comparator="<",
        threshold=0.75,
        on_stage=InvestigationStage.BLACK_BOX,
    )
    spec = _spec(abort_if=[pred])
    blob = spec.model_dump_json()
    loaded = InvestigationSpec.model_validate_json(blob)
    assert isinstance(loaded.abort_if[0], AbortPredicate)
    assert loaded.abort_if[0].predicate_id == "abort-acc-low"


def test_abort_threshold_range_validated() -> None:
    with pytest.raises(ValidationError, match="range"):
        AbortPredicate(
            predicate_id="bad",
            description="bad",
            metric=MetricName.ACCURACY,
            comparator="<",
            threshold=2.0,  # accuracy is in [0, 1]
        )


def test_abort_legacy_strings_still_accepted() -> None:
    spec = _spec(abort_if=["accuracy too low — abort"])
    assert spec.abort_if == ["accuracy too low — abort"]
    assert typed_abort_predicates(spec) == []


def test_abort_mixed_list_partitioned() -> None:
    pred = AbortPredicate(
        predicate_id="p1",
        description="d",
        metric=MetricName.LOGIT_DIFF,
        comparator="<",
        threshold=1.0,
    )
    spec = _spec(abort_if=[pred, "free-form note"])
    typed = typed_abort_predicates(spec)
    assert len(typed) == 1
    assert typed[0].predicate_id == "p1"
    assert "free-form note" in spec.abort_if


def test_legacy_pythia_spec_still_loads() -> None:
    """The pre-existing approved spec uses string abort_if — must keep loading."""
    path = Path("outputs/specs/do-identification-pythia-410m-v1_rev1.json")
    if not path.exists():
        pytest.skip(f"missing fixture: {path}")
    spec = InvestigationSpec.model_validate_json(path.read_text())
    assert all(isinstance(x, str) for x in spec.abort_if)
    assert typed_abort_predicates(spec) == []


def test_abort_predicate_describe_visible() -> None:
    """`describe_spec` should advertise AbortPredicate in the abort_if field."""
    from autointerp.spec_describe import describe_spec_markdown

    md = describe_spec_markdown()
    assert "abort_if" in md
    assert "AbortPredicate" in md


def test_abort_predicate_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        AbortPredicate.model_validate(
            {
                "predicate_id": "p",
                "description": "d",
                "metric": "accuracy",
                "comparator": "<",
                "threshold": 0.5,
                "MYSTERY": True,
            }
        )


def test_abort_predicate_with_dict_inputs() -> None:
    """Loading abort_if from JSON dicts (the agent's path) round-trips."""
    raw = {
        "predicate_id": "p",
        "description": "d",
        "metric": "logit_diff",
        "comparator": "<",
        "threshold": 1.0,
        "on_stage": "black_box",
    }
    spec = _spec(abort_if=[raw])
    assert isinstance(spec.abort_if[0], AbortPredicate)
    assert spec.abort_if[0].on_stage is InvestigationStage.BLACK_BOX


def test_typed_abort_predicates_helper_signature() -> None:
    """Sanity: helper is callable with a real spec."""
    spec = _spec(abort_if=[])
    assert typed_abort_predicates(spec) == []


def test_describe_includes_abortpredicate_section() -> None:
    """Either inline or as its own section, AbortPredicate must be discoverable."""
    from autointerp.spec_describe import describe_spec_markdown

    md = describe_spec_markdown()
    # round-trip JSON of the spec field name itself, plus reference to nested type
    assert "abort_if" in md and ("AbortPredicate" in md or "predicate_id" in md)


def test_abort_predicate_json_schema_round_trip() -> None:
    pred = AbortPredicate(
        predicate_id="abort-faith-fail",
        description="Abort if circuit faithfulness drops below 0.5",
        metric=MetricName.FAITHFULNESS,
        comparator="<",
        threshold=0.5,
        on_stage=InvestigationStage.VALIDATION,
    )
    payload = json.loads(pred.model_dump_json())
    assert payload["on_stage"] == "validation"
    restored = AbortPredicate.model_validate(payload)
    assert restored == pred
