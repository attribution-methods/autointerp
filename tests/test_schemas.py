from autointerp.schemas import (
    BehaviorSpec,
    ChatMessage,
    EvidenceStrength,
    InvestigationReport,
    MetricResult,
    ModelRef,
    PromptBatch,
    PromptCase,
)


def test_report_round_trip(tmp_path):
    behavior = BehaviorSpec(
        behavior_id="sycophancy-demo",
        description="Model agrees with users too readily.",
    )
    report = InvestigationReport(
        report_id="report-1",
        behavior=behavior,
        models=[ModelRef(model_id="fixture-model", provider="local")],
        prompt_batches=[
            PromptBatch(
                batch_id="batch-1",
                behavior_id=behavior.behavior_id,
                cases=[
                    PromptCase(
                        prompt_id="p1",
                        messages=[ChatMessage(role="user", content="Is my wrong answer right?")],
                    )
                ],
            )
        ],
        claims=["Round-trip serialization works."],
    )
    path = tmp_path / "report.json"
    report.write_json(path)

    loaded = InvestigationReport.read_json(path)
    assert loaded == report
    assert loaded.as_json_dict()["behavior"]["behavior_id"] == "sycophancy-demo"


def test_metric_result_records_threshold_status():
    metric = MetricResult(
        metric_id="heldout_hit_rate",
        value=0.75,
        threshold=0.5,
        passed=True,
    )
    assert metric.passed is True
    assert EvidenceStrength.MODERATE.value == "moderate"



# ---------------------------------------------------------------------------
# model/method coherence (white-box tools need open weights)
# ---------------------------------------------------------------------------


def _coherence_spec(model_id: str, tools):
    from autointerp.schemas import BehaviorSpec, InvestigationStage, ModelRef
    from autointerp.spec import (
        ContrastSpec,
        Criterion,
        DatasetSpec,
        InvestigationSpec,
        MetricName,
        PatternId,
        StageSpec,
    )

    return InvestigationSpec(
        spec_id="coherence-fixture",
        question="q",
        hypothesis="h",
        phenomenon_id="p",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id=model_id),
        dataset=DatasetSpec(dataset_id="d", source="generated", n_samples=10),
        contrast=ContrastSpec(
            contrast_id="c", positive_template="a", negative_template="b"
        ),
        stages=[
            StageSpec(
                stage=InvestigationStage.BLACK_BOX,
                pattern=PatternId.CUSTOM,
                tools=tools,
                metrics=[MetricName.ACCURACY],
            )
        ],
        success_criteria=[
            Criterion(
                criterion_id="c1",
                description="d",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.5,
            )
        ],
    )


def test_is_api_only_model_patterns():
    from autointerp.spec import is_api_only_model

    for closed in (
        "gpt-4", "gpt-4o", "openai/gpt-5.2", "openrouter/openai/gpt-4o",
        "claude-haiku-4-5", "anthropic/claude-sonnet-4-5", "gemini-2.0-pro",
        "o1-mini", "o3", "grok-3",
    ):
        assert is_api_only_model(closed), closed
    for open_or_unknown in (
        "gpt2", "sshleifer/tiny-gpt2", "EleutherAI/pythia-1.4b",
        "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
        "allenai/OLMo-2-7B", "facebook/opt-1.3b", "m",
    ):
        assert not is_api_only_model(open_or_unknown), open_or_unknown


def test_whitebox_tools_reject_api_only_models():
    import pytest
    from pydantic import ValidationError

    from autointerp.spec import ToolName

    for bad in ("gpt-4", "openai/gpt-4o", "claude-sonnet-4-5", "o3-mini"):
        with pytest.raises(ValidationError, match="closed-weights"):
            _coherence_spec(bad, [ToolName.ACTIVATION_PATCHING])
    # Black-box-only stages remain legal on API models.
    _coherence_spec("gpt-4", [ToolName.BLACKBOX_PROBE])
    # Open-weights (and unprovable/unknown) ids pass with white-box tools.
    _coherence_spec("gpt2", [ToolName.ACTIVATION_PATCHING])
    _coherence_spec("EleutherAI/pythia-1.4b", [ToolName.SAE_INSPECT])


# ---------------------------------------------------------------------------
# finalize guard: a stage cannot declare a metric it can't produce
# ---------------------------------------------------------------------------


def _stage_spec(stage, tools, metrics, model_id="gpt2"):
    from autointerp.schemas import BehaviorSpec, ModelRef
    from autointerp.spec import (
        ContrastSpec,
        Criterion,
        DatasetSpec,
        InvestigationSpec,
        MetricName,
        PatternId,
        StageSpec,
    )

    return InvestigationSpec(
        spec_id="finalize-guard-fixture",
        question="q", hypothesis="h", phenomenon_id="p",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id=model_id),
        dataset=DatasetSpec(dataset_id="d", source="generated", n_samples=10),
        contrast=ContrastSpec(
            contrast_id="c", positive_template="a", negative_template="b"
        ),
        stages=[StageSpec(stage=stage, pattern=PatternId.CUSTOM,
                          tools=tools, metrics=metrics)],
        success_criteria=[Criterion(
            criterion_id="c1", description="d", metric=MetricName.ACCURACY,
            comparator=">=", threshold=0.5,
        )],
    )


def test_unproducible_metric_refs_flags_causal_in_setup():
    from autointerp.schemas import InvestigationStage
    from autointerp.spec import MetricName, ToolName
    from autointerp_agent.stage0_tools import _unproducible_metric_refs

    # Causal metric in a setup stage → flagged (this is the real deadlock).
    bad = _stage_spec(
        InvestigationStage.SETUP, [ToolName.ACTIVATION_CACHE],
        [MetricName.PATCH_EFFECT_RECOVERY],
    )
    issues = _unproducible_metric_refs(bad)
    assert len(issues) == 1 and "patch_effect_recovery" in issues[0]

    # Same causal metric in an intervention stage → fine.
    ok = _stage_spec(
        InvestigationStage.INTERVENTION, [ToolName.ACTIVATION_PATCHING],
        [MetricName.PATCH_EFFECT_RECOVERY],
    )
    assert _unproducible_metric_refs(ok) == []

    # A behavioral metric in a setup stage → fine.
    ok2 = _stage_spec(
        InvestigationStage.SETUP, [ToolName.ACTIVATION_CACHE],
        [MetricName.LOGIT_DIFF],
    )
    assert _unproducible_metric_refs(ok2) == []


# ---------------------------------------------------------------------------
# list_metrics / read_metric surface runtime implementation status
# ---------------------------------------------------------------------------


def test_list_metrics_separates_runnable_from_reserved():
    import asyncio

    from autointerp.pipelines.investigation.metrics import REGISTRY
    from autointerp_agent.stage0_tools import _list_metrics

    out, ok = asyncio.run(_list_metrics({}))
    assert ok
    runnable_part, _, reserved_part = out.partition("NOT YET RUNNABLE")
    # Every implemented metric is advertised as runnable…
    for m in REGISTRY:
        assert m.value in runnable_part
    # …and declared-but-unimplemented ones are clearly flagged, not offered.
    for name in ("feature_activation_density", "mutual_information",
                 "causal_indirect_effect", "paraphrase_invariance"):
        assert name in reserved_part
        assert name not in runnable_part


def test_read_metric_reports_runnable_flag():
    import asyncio

    from autointerp_agent.stage0_tools import _read_metric

    runnable_out, _ = asyncio.run(_read_metric({"name": "logit_diff"}))
    assert "runnable: yes" in runnable_out
    reserved_out, _ = asyncio.run(_read_metric({"name": "mutual_information"}))
    assert "runnable: NO" in reserved_out
