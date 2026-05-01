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

