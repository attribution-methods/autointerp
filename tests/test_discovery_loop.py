"""Tests for the hill-climbing discovery sub-agent.

Covers (a) the AUC-K metric impls, (b) the end-to-end dry-run discovery loop,
(c) the generic run_hillclimb engine with a non-discovery task, and (d)
seed-variance + archive bookkeeping. All run with no model, GPU, or LLM creds.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autointerp.pipelines.investigation.discovery import (
    HillClimbConfig,
    HillClimbTask,
    run_discovery_subagent,
    run_hillclimb,
)
from autointerp.pipelines.investigation.discovery.algorithm_template import (
    __file__ as _template_file,
)
from autointerp.pipelines.investigation.metrics import REGISTRY, MetricRegistryError
from autointerp.spec import METRIC_META, MetricName

# ---- AUC-K metrics ---------------------------------------------------------


def _compute(name: MetricName, inputs: dict) -> float:
    return REGISTRY[name].compute(inputs)


def test_mean_auc_k_perfect_curve() -> None:
    out = _compute(
        MetricName.MEAN_ABLATION_AUC_K,
        {"delta_curve_per_pair": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]},
    )
    assert out == pytest.approx(1.0)


def test_mean_auc_k_linear_ramp() -> None:
    out = _compute(
        MetricName.MEAN_STEERING_AUC_K,
        {"delta_curve_per_pair": [[0.0, 0.5, 1.0]]},
    )
    assert out == pytest.approx(0.5)


def test_mean_auc_k_rejects_out_of_range() -> None:
    with pytest.raises(MetricRegistryError):
        _compute(
            MetricName.MEAN_ABLATION_AUC_K,
            {"delta_curve_per_pair": [[0.0, 1.5]]},
        )


def test_combined_auc_k_from_scalars() -> None:
    out = _compute(
        MetricName.COMBINED_AUC_K,
        {"mean_ablation_auc_k": 0.8, "mean_steering_auc_k": 0.6},
    )
    assert out == pytest.approx(0.7)


def test_combined_auc_k_from_curves() -> None:
    out = _compute(
        MetricName.COMBINED_AUC_K,
        {
            "ablation_delta_curve_per_pair": [[1.0, 1.0]],
            "steering_delta_curve_per_pair": [[0.0, 1.0]],
        },
    )
    assert out == pytest.approx(0.75)


def test_auc_metrics_registered_and_metaed() -> None:
    for name in (
        MetricName.MEAN_ABLATION_AUC_K,
        MetricName.MEAN_STEERING_AUC_K,
        MetricName.COMBINED_AUC_K,
    ):
        assert name in REGISTRY
        assert name in METRIC_META
        assert METRIC_META[name].value_range == (0.0, 1.0)


# ---- dry-run discovery loop ------------------------------------------------


def test_dry_run_loop_produces_best_candidate(tmp_path) -> None:
    result = asyncio.run(
        run_discovery_subagent(
            session_dir=tmp_path / "disco",
            task="rank attention heads for the dry-run behavior",
            dry_run=True,
        )
    )
    assert result.best_candidate == "cand_000"
    assert result.best_reward is not None and 0.0 <= result.best_reward <= 1.0
    assert result.iterations_run == 1
    assert result.best_summary["objective_metric"] == "combined_auc_k"
    assert (result.session_dir / "cand_000.py").exists()
    assert (result.session_dir / "results" / "cand_000.json").exists()
    assert (result.session_dir / "archive.jsonl").exists()
    # Archive carries the best candidate.
    assert result.archive and result.archive[0]["candidate"] == "cand_000"


def test_dry_run_seed_variance(tmp_path) -> None:
    # Multiple seeds → the best candidate is re-evaluated and a std reported.
    result = asyncio.run(
        run_discovery_subagent(
            session_dir=tmp_path / "disco",
            task="rank heads",
            seeds=[0, 1, 2, 3],
            dry_run=True,
        )
    )
    assert result.best_reward_std is not None
    assert result.best_reward_std >= 0.0


# ---- generic engine (non-discovery task) -----------------------------------


def test_run_hillclimb_generic_task(tmp_path) -> None:
    from pathlib import Path

    task = HillClimbTask(
        name="generic_demo",
        task_text="optimize a ranking artifact (generic engine smoke test)",
        reward_metric="combined_auc_k",
        reward_description="balance ablation + steering AUC.",
        candidate_template=Path(_template_file),
        contract_instructions="define score(...) -> list[Candidate].",
        artifact_suffix=".py",
    )
    result = asyncio.run(
        run_hillclimb(
            session_dir=tmp_path / "generic",
            task=task,
            config=HillClimbConfig(dry_run=True),
        )
    )
    assert result.best_candidate == "cand_000"
    assert result.best_reward is not None
    # cost tracker persisted, archive readable.
    assert (result.session_dir / "cost.json").exists()
    archive = (result.session_dir / "archive.jsonl").read_text().strip()
    assert json.loads(archive.splitlines()[0])["candidate"] == "cand_000"


def test_dry_run_no_token_spend(tmp_path) -> None:
    # Dry-run makes no LLM calls, so no budget is consumed.
    result = asyncio.run(
        run_discovery_subagent(
            session_dir=tmp_path / "disco", task="rank heads", dry_run=True
        )
    )
    assert result.tokens_used == 0
    assert result.cost_usd == 0.0


# ---- Tier-2 discover_features handler (gating, budget, report) --------------


def _approved_discovery_spec(tmp_path, *, with_tool: bool = True):
    from datetime import datetime, timezone

    from autointerp.schemas import InvestigationStage
    from autointerp.spec import (
        Approval,
        BehaviorSpec,
        Budget,
        ContrastSpec,
        Criterion,
        DatasetSpec,
        DiscoveryConfig,
        InvestigationSpec,
        ModelRef,
        PatternId,
        SpecStatus,
        StageSpec,
        ToolName,
    )

    tools = [ToolName.SAE_INSPECT]
    if with_tool:
        tools = [ToolName.DISCOVER_FEATURES, ToolName.SAE_INSPECT]
    spec = InvestigationSpec(
        spec_id="disco-test", revision=1, question="q", hypothesis="h",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="gpt2"),
        dataset=DatasetSpec(dataset_id="ds", source="generated", n_samples=10,
                            split="dev", seed=1),
        contrast=ContrastSpec(contrast_id="c", positive_template="{x}",
                              negative_template="{y}", pairing="matched"),
        stages=[StageSpec(
            stage=InvestigationStage.FEATURE_ANALYSIS, pattern=PatternId.PROBE_THEN_SAE,
            tools=tools, metrics=[MetricName.COMBINED_AUC_K],
            discovery=DiscoveryConfig(seeds=[0, 1], top_k=3) if with_tool else None,
        )],
        success_criteria=[Criterion(criterion_id="c1", description="d",
            metric=MetricName.COMBINED_AUC_K, comparator=">=", threshold=0.5,
            on_split="heldout")],
        budget=Budget(max_tokens=1_000_000), status=SpecStatus.APPROVED,
        approval=Approval(approver="t", approver_kind="agent",
                          approved_at=datetime.now(timezone.utc)),
    )
    return spec


def _handler(handle, name):
    from autointerp.pipelines.investigation.tools import create_investigation_tools

    return {t.name: t for t in create_investigation_tools(handle)}[name].handler


def test_handler_dry_run_happy_path(tmp_path) -> None:
    from autointerp.pipelines.investigation.report import assemble_report
    from autointerp.pipelines.investigation.run_dir import init_run

    spec = _approved_discovery_spec(tmp_path)
    handle = init_run(spec, runs_root=tmp_path / "runs")
    out, ok = asyncio.run(_handler(handle, "discover_features")(
        {"task": "rank SAE features", "dry_run": True}))
    assert ok
    res = json.loads(out)
    assert res["best_candidate"] == "cand_000"
    assert res["best_reward_std"] is not None  # seeds=[0,1] → variance computed
    # Report surfaces the (advisory) session.
    sessions = assemble_report(handle).metadata.get("discovery_sessions")
    assert sessions and sessions[0]["best_candidate"] == "cand_000"


def test_handler_refuses_when_stage_lacks_tool(tmp_path) -> None:
    from autointerp.pipelines.investigation.run_dir import init_run

    spec = _approved_discovery_spec(tmp_path, with_tool=False)
    handle = init_run(spec, runs_root=tmp_path / "runs")
    out, ok = asyncio.run(_handler(handle, "discover_features")(
        {"task": "rank SAE features", "dry_run": True}))
    assert not ok
    assert "does not list `discover_features`" in out


# ---- memory: bounded context + structured lookup --------------------------


def _archive(n: int) -> list[dict]:
    return [
        {"candidate": f"cand_{i:03d}", "reward": 0.9 - i * 0.1,
         "code": f"# program {i}\ndef score():\n    return {i}\n"}
        for i in range(n)
    ]


def test_prompt_inlines_only_parent_plus_n(tmp_path) -> None:
    from pathlib import Path

    from autointerp.pipelines.investigation.discovery.engine import _build_user_prompt
    from autointerp.pipelines.investigation.discovery.task import HillClimbTask

    arch = _archive(5)  # 5 candidates in the archive
    task = HillClimbTask(
        name="t", task_text="rank", reward_metric="combined_auc_k",
        reward_description="", candidate_template=Path("x"),
        contract_instructions="c",
    )
    prompt = _build_user_prompt(
        task=task, parent=arch[0], archive=arch, log=[],
        iteration=2, n_inline=1, agentic=False, session_dir=tmp_path,
    )
    # All 5 appear in the compact leaderboard...
    for e in arch:
        assert e["candidate"] in prompt
    # ...but only parent + 1 extra have FULL code inlined (2 code fences).
    assert prompt.count("```python") == 2
    # Single mode: no structured-lookup pointer.
    assert "Structured memory" not in prompt


def test_prompt_agentic_points_to_disk(tmp_path) -> None:
    from pathlib import Path

    from autointerp.pipelines.investigation.discovery.engine import _build_user_prompt
    from autointerp.pipelines.investigation.discovery.task import HillClimbTask

    arch = _archive(4)
    task = HillClimbTask(
        name="t", task_text="rank", reward_metric="combined_auc_k",
        reward_description="", candidate_template=Path("x"),
        contract_instructions="c",
    )
    prompt = _build_user_prompt(
        task=task, parent=arch[0], archive=arch, log=[],
        iteration=2, n_inline=0, agentic=True, session_dir=tmp_path,
    )
    # n_inline=0 → only the parent's code is inlined.
    assert prompt.count("```python") == 1
    # Agentic mode points at the structured store for on-demand lookup.
    assert "archive.jsonl" in prompt and "Structured memory" in prompt


def test_insights_summary_is_deterministic() -> None:
    from autointerp.pipelines.investigation.discovery.engine import _insights

    log = [
        {"candidate": "cand_000", "reward": 0.5},
        {"candidate": "cand_001", "status": "eval_failed"},
        {"candidate": "cand_002", "reward": 0.8},
        {"candidate": "cand_003", "reward": 0.7},
    ]
    summary = _insights(log)
    assert "best=0.8000 (cand_002)" in summary
    assert "1 successful attempts since best" in summary
    assert "eval_failed×1" in summary
