"""Tests for the commit_artifact gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import (
    AblationFlags,
    ArtifactGateError,
    commit_artifact,
    init_run,
    write_report,
)
from autointerp.pipelines.investigation.metrics import (
    MetricRegistryError,
    _require_keys,
)
from autointerp.pipelines.investigation.state import (
    CriterionRecord,
    ProvenanceToken,
    StageStatus,
    TerminalState,
    Verdict,
    now_iso,
    read_state,
    write_state,
)
from autointerp.pipelines.investigation.tools import (
    _did_any_empirical_work,
    _resolve_metric_inputs,
    create_investigation_tools,
)
from autointerp.schemas import (
    ActivationCacheRef,
    BehavioralFinding,
    BehaviorSpec,
    ChatMessage,
    EvidenceStrength,
    MetricResult,
    ModelRef,
    PromptBatch,
    PromptCase,
)
from autointerp.schemas import ModelRef as SchemaModelRef
from autointerp.spec import (
    Approval,
    Budget,
    ContrastSpec,
    Criterion,
    DatasetSpec,
    InvestigationSpec,
    InvestigationStage,
    MetricName,
    PatternId,
    SpecStatus,
    StageSpec,
    ToolName,
)


def _approved_spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="artifacts-fixture",
        revision=1,
        question="Q",
        hypothesis="H",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="fixture/model", provider="local"),
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
                criterion_id="c1",
                description="d",
                metric=MetricName.ACCURACY,
                comparator=">=",
                threshold=0.5,
                on_split="dev",
            )
        ],
        budget=Budget(max_tool_calls=10),
        status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent", approved_at=now_iso()),
    )


def _make_prompt_batch(batch_id: str = "b1", split: str = "dev") -> PromptBatch:
    return PromptBatch(
        batch_id=batch_id,
        behavior_id="b",
        cases=[
            PromptCase(prompt_id="p1", messages=[ChatMessage(role="user", content="hi")]),
        ],
        split=split,
    )


def _seed_token(
    handle, *, token: str, metric: str, metric_id: str, value: float, stage_idx: int = 0
):
    state = read_state(handle.state_path)
    state.pending_provenance_tokens[token] = ProvenanceToken(
        metric=metric,
        metric_id=metric_id,
        value=value,
        inputs_hash="sha256:test",
        issued_at=now_iso(),
        stage_idx=stage_idx,
    )
    write_state(handle.state_path, state)


# ---- happy paths -----------------------------------------------------------


def test_commit_prompt_batch_roundtrip(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    ref = commit_artifact(handle, "PromptBatch", batch)
    assert ref.kind == "PromptBatch"
    assert ref.artifact_id == "b1"
    assert ref.relpath == "prompt_batches/b1.json"
    assert ref.path.exists()
    payload = json.loads(ref.path.read_text())
    assert payload["batch_id"] == "b1"
    assert payload["metadata"]["_provenance"]["stage_idx"] == 0
    assert payload["metadata"]["_provenance"]["split"] == "dev"

    state = read_state(handle.state_path)
    assert state.stage_status["0"].status is StageStatus.IN_PROGRESS
    assert state.stage_status["0"].artifact_refs == ["prompt_batches/b1.json"]
    assert state.budget_consumed.tool_calls == 1


def test_commit_dict_payload_validated(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    ref = commit_artifact(
        handle,
        "PromptBatch",
        {
            "batch_id": "b2",
            "behavior_id": "b",
            "cases": [{"prompt_id": "p1", "messages": [{"role": "user", "content": "hi"}]}],
            "split": "dev",
        },
    )
    assert ref.path.exists()


def test_commit_finding_into_stage_dir(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    finding = BehavioralFinding(
        finding_id="signal-check",
        behavior_id="b",
        summary="signal exists",
        evidence_strength=EvidenceStrength.MODERATE,
    )
    ref = commit_artifact(handle, "BehavioralFinding", finding, split="dev")
    assert ref.relpath == "findings/stage_0_black_box/behavioral_signal-check.json"
    assert ref.path.exists()


def test_commit_activation_cache_ref(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    cache = ActivationCacheRef(
        cache_id="cache-1",
        model=SchemaModelRef(model_id="m"),
        source_path="activations/cache-1/",
        prompt_batch_id="b1",
        layers=[0, 1, 2],
    )
    ref = commit_artifact(handle, "ActivationCacheRef", cache, split="dev")
    assert ref.relpath == "activations/cache-1.json"


def test_commit_metric_with_valid_token(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-001", value=0.83)
    ref = commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")
    assert ref.relpath == "findings/stage_0_black_box/metric_acc-001.json"
    state = read_state(handle.state_path)
    assert "tok1" not in state.pending_provenance_tokens
    assert state.provenance_tokens_consumed == 1


# ---- gate failures ---------------------------------------------------------


def test_unknown_kind_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(ArtifactGateError, match="Unknown artifact kind"):
        commit_artifact(handle, "Whatever", {}, split="dev")


def test_invalid_payload_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    with pytest.raises(ArtifactGateError, match="failed validation"):
        commit_artifact(handle, "PromptBatch", {"batch_id": "x"})


def test_metric_payload_with_split_inside_gets_actionable_hint(tmp_path: Path) -> None:
    """The most common weak-agent mistake: `split` crammed into the MetricResult
    payload (it is a top-level arg). The gate must reject AND tell the agent how
    to fix it, including the one-shot tool, so it doesn't loop for many turns."""
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    bad_payload = {
        "metric_id": "logit_diff_001", "value": 1.1, "passed": True,
        "split": "dev",  # <- belongs as a sibling arg, not in payload
    }
    with pytest.raises(ArtifactGateError) as ei:
        commit_artifact(handle, "MetricResult", bad_payload,
                        split="dev", provenance_token="tok1")
    msg = str(ei.value)
    assert "split is a top-level argument" in msg
    assert "compute_and_commit_metric" in msg  # points at the one-shot path


def test_invented_artifact_fields_get_the_real_schema(tmp_path: Path) -> None:
    """A real Stage-3 bail: the agent INVENTED InterventionResult fields
    (patches/prompts_used/status) and omitted the required ones, hit a raw
    pydantic dump it couldn't decode, and requested a needless spec revision.
    The gate must now print the artifact's real schema and name the bad fields,
    so the agent self-corrects in one retry instead of bailing."""
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    bad = {"patches": [{"layer": 24}], "prompts_used": [], "status": "executed"}
    with pytest.raises(ArtifactGateError) as ei:
        commit_artifact(handle, "InterventionResult", bad, split="dev")
    msg = str(ei.value)
    assert "intervention_id" in msg and "method" in msg  # required fields shown
    assert "delta" in msg and "metadata" in msg          # optional fields shown
    assert "Unknown fields you sent: patches, prompts_used, status" in msg
    assert "NOT a reason to request_spec_revision" in msg


def test_split_required_for_non_promptbatch(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    finding = BehavioralFinding(finding_id="f", behavior_id="b", summary="s")
    with pytest.raises(ArtifactGateError, match="split is required"):
        commit_artifact(handle, "BehavioralFinding", finding)


def test_promptbatch_split_arg_must_match(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch(split="dev")
    with pytest.raises(ArtifactGateError, match="split mismatch"):
        commit_artifact(handle, "PromptBatch", batch, split="heldout")


def test_overwrite_refused(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    commit_artifact(handle, "PromptBatch", batch)
    with pytest.raises(ArtifactGateError, match="append-only"):
        commit_artifact(handle, "PromptBatch", batch)


def test_metric_requires_token(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    metric = MetricResult(metric_id="acc-001", value=0.83)
    with pytest.raises(ArtifactGateError, match="provenance_token"):
        commit_artifact(handle, "MetricResult", metric, split="dev")


def test_token_not_reusable(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-001", value=0.83)
    commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")
    metric2 = MetricResult(metric_id="acc-002", value=0.83)
    with pytest.raises(ArtifactGateError, match="unknown or already consumed"):
        commit_artifact(handle, "MetricResult", metric2, split="dev", provenance_token="tok1")


def test_token_metric_id_mismatch(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-OTHER", value=0.83)
    with pytest.raises(ArtifactGateError, match="metric_id"):
        commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")


def test_token_value_mismatch_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83)
    metric = MetricResult(metric_id="acc-001", value=0.99)
    with pytest.raises(ArtifactGateError, match="value"):
        commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")


def test_token_stage_mismatch_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    _seed_token(
        handle, token="tok1", metric="accuracy", metric_id="acc-001", value=0.83, stage_idx=1
    )
    metric = MetricResult(metric_id="acc-001", value=0.83)
    with pytest.raises(ArtifactGateError, match="stage_idx"):
        commit_artifact(handle, "MetricResult", metric, split="dev", provenance_token="tok1")


def test_token_only_for_metric_result(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    with pytest.raises(ArtifactGateError, match="only valid for MetricResult"):
        commit_artifact(handle, "PromptBatch", batch, provenance_token="tok-anything")


def test_terminal_state_blocks_commits(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    state = read_state(handle.state_path)
    state.terminal_state = TerminalState.COMPLETED
    write_state(handle.state_path, state)
    with pytest.raises(ArtifactGateError, match="terminal state"):
        commit_artifact(handle, "PromptBatch", _make_prompt_batch())


def test_reserved_provenance_metadata_rejected(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    batch = _make_prompt_batch()
    batch = batch.model_copy(update={"metadata": {"_provenance": {"hijack": True}}})
    with pytest.raises(ArtifactGateError, match="_provenance"):
        commit_artifact(handle, "PromptBatch", batch)


def test_log_jsonl_appended(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    commit_artifact(handle, "PromptBatch", _make_prompt_batch())
    lines = [
        json.loads(line)
        for line in handle.log_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["tool"] == "commit_artifact"
    assert entry["args"]["kind"] == "PromptBatch"
    assert entry["ok"] is True
    assert entry["budget_after"]["tool_calls"] == 1


def test_in_progress_started_at_set_once(tmp_path: Path) -> None:
    spec = _approved_spec()
    handle = init_run(spec, runs_root=tmp_path)
    commit_artifact(handle, "PromptBatch", _make_prompt_batch("b1"))
    state1 = read_state(handle.state_path)
    started1 = state1.stage_status["0"].started_at
    assert started1 is not None
    commit_artifact(handle, "PromptBatch", _make_prompt_batch("b2"))
    state2 = read_state(handle.state_path)
    assert state2.stage_status["0"].started_at == started1
    assert state2.stage_status["0"].status is StageStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# request_spec_revision: don't let a weak agent bail before doing any work
# ---------------------------------------------------------------------------


def _tool(handle, name):
    return next(t for t in create_investigation_tools(handle) if t.name == name)


def test_did_any_empirical_work_detects_outputs(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path)
    assert _did_any_empirical_work(handle) is False  # fresh run, nothing done
    handle.scratch_dir.mkdir(parents=True, exist_ok=True)
    (handle.scratch_dir / "logits.json").write_text("[]")
    assert _did_any_empirical_work(handle) is True  # produced output


def test_request_revision_rejected_before_any_work(tmp_path: Path) -> None:
    """The exact failure from the field: the agent calls request_spec_revision
    as its first move because logit_diff's inputs 'aren't available' — when
    producing them is its job. The gate rejects and teaches, and the run is NOT
    flipped to a terminal revision state."""
    handle = init_run(_approved_spec(), runs_root=tmp_path)
    revise = _tool(handle, "request_spec_revision")
    out, ok = asyncio.run(revise.handler(
        {"reason": "logit_diff needs target_logits/foil_logits that aren't here"}
    ))
    assert ok is False
    assert "not run a single experiment" in out
    assert "DATA YOU PRODUCE" in out
    assert read_state(handle.state_path).terminal_state is None


def test_request_revision_allowed_after_real_work(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path)
    handle.scratch_dir.mkdir(parents=True, exist_ok=True)
    (handle.scratch_dir / "probe_out.json").write_text("{}")  # genuine attempt
    revise = _tool(handle, "request_spec_revision")
    out, ok = asyncio.run(revise.handler(
        {"reason": "after running probes, the dataset lacks the needed contrast"}
    ))
    assert ok is True
    assert read_state(handle.state_path).terminal_state is TerminalState.REVISION_REQUESTED


def test_request_revision_rejected_when_all_criteria_evaluated(tmp_path: Path) -> None:
    """The real run's stall: the sole criterion already PASSED, but a stage-gate
    snag pushed the agent into request_spec_revision — stranding an otherwise
    complete run as INCOMPLETE. With every criterion evaluated the verdict is in,
    so the revision is refused and the agent is told to advance_stage to finish.
    The run must NOT be flipped to a terminal revision state."""
    handle = init_run(_approved_spec(), runs_root=tmp_path)
    _mark_criterion_passed(handle)  # the spec's only criterion (c1) now PASSED
    revise = _tool(handle, "request_spec_revision")
    out, ok = asyncio.run(revise.handler(
        {"reason": "stage 0 gate shows custom not committed due to artifact naming"}
    ))
    assert ok is False
    assert "verdict" in out.lower() and "advance_stage" in out
    assert read_state(handle.state_path).terminal_state is None


def test_metric_missing_inputs_error_points_to_running_the_model() -> None:
    with pytest.raises(MetricRegistryError) as ei:
        _require_keys(
            MetricName.LOGIT_DIFF,
            {"dataset": "ds", "model_id": "gpt2", "seed": 1},
            ["target_logits", "foil_logits"],
        )
    msg = str(ei.value)
    assert "running the model" in msg
    assert "metadata" in msg
    assert "not a reason to request a spec revision" in msg.lower()


# ---------------------------------------------------------------------------
# grounding guard: a run that never ran the model is flagged as fabricated
# ---------------------------------------------------------------------------


def _mark_criterion_passed(handle) -> None:
    """Seed an evaluated criterion in state — the grounding check keys off
    state.criteria_evaluated, so this is all the setup it needs."""
    state = read_state(handle.state_path)
    state.criteria_evaluated["c1"] = CriterionRecord(
        verdict=Verdict.PASS, value=0.83, metric="accuracy", comparator=">=",
        threshold=0.5,
        metric_result_ref="findings/stage_0_black_box/metric_acc-001.json",
        evaluated_at=now_iso(),
    )
    write_state(handle.state_path, state)


def test_report_flags_run_with_no_model_execution(tmp_path: Path) -> None:
    """The fabrication case from the field: a metric is committed and the
    criterion 'passes', but no prompt batches / activations / generations /
    scripts exist — the model was never run. The report flags it."""
    handle = init_run(_approved_spec(), runs_root=tmp_path)
    _mark_criterion_passed(handle)
    write_report(handle)
    report = json.loads((handle.root / "report.json").read_text())
    warning = report["metadata"]["grounding_warning"]
    assert warning is not None and "NOT GROUNDED" in warning


def test_report_not_flagged_when_model_was_queried(tmp_path: Path) -> None:
    # Coarse grounding (v1): a committed prompt batch is "the model was used".
    handle = init_run(_approved_spec(), runs_root=tmp_path, flags=_V1)
    _mark_criterion_passed(handle)
    commit_artifact(handle, "PromptBatch", _make_prompt_batch())  # real interaction
    write_report(handle)
    report = json.loads((handle.root / "report.json").read_text())
    assert report["metadata"]["grounding_warning"] is None


def test_report_not_flagged_when_scripts_written(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path, flags=_V1)
    _mark_criterion_passed(handle)
    handle.scripts_dir.mkdir(parents=True, exist_ok=True)
    (handle.scripts_dir / "run_model.py").write_text("import torch  # ran the model")
    write_report(handle)
    report = json.loads((handle.root / "report.json").read_text())
    assert report["metadata"]["grounding_warning"] is None


# ---------------------------------------------------------------------------
# input provenance: metric inputs must come from a produced file
# ---------------------------------------------------------------------------


# v1 = file sourcing only (input-provenance v2 capture requirement off).
_V1 = AblationFlags(require_captured_inputs=False)
# fully relaxed inputs (inline allowed) — for tests not about provenance.
_OPEN = AblationFlags(require_captured_inputs=False, require_sourced_inputs=False)


def _write_inputs(handle, name, obj) -> str:
    handle.scratch_dir.mkdir(parents=True, exist_ok=True)
    (handle.scratch_dir / name).write_text(json.dumps(obj))
    return f"scratch/{name}"


def _capture(handle, name, obj, *, source="model_forward") -> str:
    from autointerp.tools.provenance import record_to
    return record_to(handle.root, name, obj, source=source, model_id="gpt2")


def test_compute_handler_rejects_inline_inputs(tmp_path: Path) -> None:
    """The fabrication vector — a hand-typed target_logits=[2.3]. With v1
    sourcing required, the agent's metric tool rejects inline inputs."""
    handle = init_run(_approved_spec(), runs_root=tmp_path, flags=_V1)
    cc = _tool(handle, "compute_and_commit_metric")
    out, ok = asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1",
        "inputs": {"predictions": [1, 1], "labels": [1, 1]}, "split": "dev",
    }))
    assert ok is False
    assert "must come from a file" in out


def test_compute_handler_records_file_provenance(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path, flags=_V1)
    ref = _write_inputs(handle, "acc.json", {"predictions": [1, 1], "labels": [1, 1]})
    cc = _tool(handle, "compute_and_commit_metric")
    out, ok = asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1", "inputs": ref, "split": "dev",
    }))
    assert ok is True, out
    mr = json.loads(
        (handle.root / "findings" / "stage_0_black_box" / "metric_a1.json").read_text()
    )
    prov = mr["metadata"]["input_provenance"]
    assert prov["source"] == "file"
    assert prov["ref"] == "scratch/acc.json"
    assert prov["sha256"].startswith("sha256:")


def test_inline_inputs_allowed_when_flag_off(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path, flags=_OPEN)
    cc = _tool(handle, "compute_and_commit_metric")
    out, ok = asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1",
        "inputs": {"predictions": [1, 1], "labels": [1, 1]}, "split": "dev",
    }))
    assert ok is True, out
    mr = json.loads(
        (handle.root / "findings" / "stage_0_black_box" / "metric_a1.json").read_text()
    )
    assert mr["metadata"]["input_provenance"]["source"] == "inline"


def test_report_records_metric_input_sources(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path, flags=_V1)
    ref = _write_inputs(handle, "acc.json", {"predictions": [1, 1], "labels": [1, 1]})
    cc = _tool(handle, "compute_and_commit_metric")
    asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1", "inputs": ref, "split": "dev",
    }))
    write_report(handle)
    report = json.loads((handle.root / "report.json").read_text())
    src = report["metadata"]["metric_input_sources"]["a1"]
    assert src["source"] == "file" and src["ref"] == "scratch/acc.json"


# ---------------------------------------------------------------------------
# input provenance v2: metric inputs must be a model_forward capture
# ---------------------------------------------------------------------------


def test_compute_handler_accepts_model_forward_capture(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path)  # v2 on by default
    ref = _capture(handle, "acc_inputs", {"predictions": [1, 1], "labels": [1, 1]})
    cc = _tool(handle, "compute_and_commit_metric")
    out, ok = asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1", "inputs": ref, "split": "dev",
    }))
    assert ok is True, out
    mr = json.loads(
        (handle.root / "findings" / "stage_0_black_box" / "metric_a1.json").read_text()
    )
    prov = mr["metadata"]["input_provenance"]
    assert prov["source"] == "capture"
    assert prov["capture_source"] == "model_forward"
    assert prov["model_id"] == "gpt2"


def test_compute_handler_rejects_plain_file_under_v2(tmp_path: Path) -> None:
    """v2: a plain produced file is no longer enough — it must be a capture."""
    handle = init_run(_approved_spec(), runs_root=tmp_path)  # v2 on
    ref = _write_inputs(handle, "acc.json", {"predictions": [1, 1], "labels": [1, 1]})
    cc = _tool(handle, "compute_and_commit_metric")
    out, ok = asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1", "inputs": ref, "split": "dev",
    }))
    assert ok is False
    assert "model_forward capture" in out


def test_compute_handler_rejects_manual_capture_under_v2(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path)
    ref = _capture(handle, "x", {"predictions": [1], "labels": [1]}, source="manual")
    cc = _tool(handle, "compute_and_commit_metric")
    out, ok = asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1", "inputs": ref, "split": "dev",
    }))
    assert ok is False  # a manual capture is not a real measurement
    assert "model_forward capture" in out


def test_report_grounded_when_criteria_backed_by_captures(tmp_path: Path) -> None:
    handle = init_run(_approved_spec(), runs_root=tmp_path)
    ref = _capture(handle, "acc_inputs", {"predictions": [1, 1], "labels": [1, 1]})
    cc = _tool(handle, "compute_and_commit_metric")
    asyncio.run(cc.handler({
        "metric": "accuracy", "metric_id": "a1", "inputs": ref, "split": "dev",
        "threshold": 0.5, "comparator": ">=", "criterion_id": "c1",
    }))
    write_report(handle)
    report = json.loads((handle.root / "report.json").read_text())
    assert report["metadata"]["grounding_warning"] is None  # traces to a real run


def test_resolve_metric_inputs_finds_cwd_relative_file(tmp_path: Path, monkeypatch) -> None:
    """A bash heredoc writes inputs to ./scratch (the working dir), not the run
    dir. The resolver must find it there too, instead of forcing the agent to
    `cp` files into the run dir (the field 'inputs file not found' loop)."""
    monkeypatch.chdir(tmp_path)
    handle = init_run(_approved_spec(), runs_root=tmp_path / "runs")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "inp.json").write_text(
        json.dumps({"predictions": [1], "labels": [1]})
    )
    inputs, prov, err = _resolve_metric_inputs(handle, "scratch/inp.json")
    assert err is None
    assert inputs == {"predictions": [1], "labels": [1]}
    assert prov["source"] == "file"
