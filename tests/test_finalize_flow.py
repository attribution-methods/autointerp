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


def test_finalize_spec_defaults_approver_to_human_user(monkeypatch, tmp_path: Path) -> None:
    """finalize_spec needs NO approver from the agent — the approver is the human
    who said 'approve'. Calling it with no args must succeed (not the old cryptic
    'approver is required', which made the agent ask the user for an 'approver
    name / human vs agent')."""
    monkeypatch.chdir(tmp_path)
    _load(_draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    ))
    try:
        # the tool no longer marks approver as required → the LLM won't ask
        assert "approver" not in s0._finalize_spec_tool().parameters.get("required", [])
        msg, ok = asyncio.run(s0._finalize_spec({}))  # no approver/kind at all
        assert ok is True and "approver is required" not in msg
        saved = json.loads(
            (tmp_path / "outputs" / "specs" / "surprise-flow_rev1.json").read_text()
        )
        assert saved["approval"]["approver"] == "user"
        assert saved["approval"]["approver_kind"] == "human"
    finally:
        s0.reset_stage0_state()


def test_update_spec_coerces_common_shape_mistakes(monkeypatch, tmp_path: Path) -> None:
    """The shapes weak models routinely botch now JUST WORK instead of erroring:
    a bare string for behavior/model is wrapped, a stray key is dropped, and a
    double-wrapped patch is unwrapped. Reproduces the real run's behavior='a
    sentence', model={'source': 'local', 'model_id': 'gpt2'}, and the
    update_spec(patch={'patch': {...}}) double-wrap."""
    monkeypatch.chdir(tmp_path)
    try:
        s0.reset_stage0_state()
        # bare string behavior + model are coerced into the right objects
        _, ok = asyncio.run(s0._update_spec({"patch": {
            "behavior": "the model raises surprisal at unexpected next tokens",
            "model": "gpt2",
        }}))
        assert ok
        data = s0.current_partial().data
        assert data["behavior"]["description"].startswith("the model")
        assert data["behavior"]["behavior_id"]  # filled in
        assert data["model"] == {"model_id": "gpt2"}

        # a stray 'source' key on model is trimmed (still valid → no error)
        _, ok2 = asyncio.run(s0._update_spec({"patch": {
            "model": {"source": "local", "model_id": "distilgpt2"},
        }}))
        assert ok2 and s0.current_partial().data["model"] == {"model_id": "distilgpt2"}

        # the double-wrap update_spec(patch={'patch': {...}}) is unwrapped
        _, ok3 = asyncio.run(s0._update_spec({"patch": {
            "patch": {"question": "How is surprise represented?"},
        }}))
        assert ok3 and s0.current_partial().data["question"].startswith("How is")
    finally:
        s0.reset_stage0_state()


def test_update_spec_errors_with_example_on_unfixable_shape(monkeypatch, tmp_path) -> None:
    """A genuinely-broken typed field (a stage missing its required keys) still
    errors — but with a COPYABLE example using real enum values, so the weak
    driver can fix `stages[5]` instead of looping on it."""
    monkeypatch.chdir(tmp_path)
    try:
        s0.reset_stage0_state()
        msg, ok = asyncio.run(s0._update_spec({"patch": {
            "stages": [{"name": "Reporting"}],  # wrong field name, missing required
        }}))
        assert ok is False
        assert "stages[0]" in msg
        assert "missing required field" in msg and "stage" in msg  # the real cause
        # a concrete, valid example (real enum values) to copy
        assert "blackbox_then_patching" in msg and "blackbox_probe" in msg
        assert "do NOT show this error to the user" in msg
    finally:
        s0.reset_stage0_state()


def test_update_spec_names_real_reason_not_generic_shape(monkeypatch, tmp_path) -> None:
    """The bug behind a real loop: a SEMANTIC validator error (comparator '<='
    on the higher-is-better metric logit_diff) was reported as 'wrong shape —
    required keys…', so the agent re-sent the same keys forever. It must now name
    the ACTUAL reason."""
    monkeypatch.chdir(tmp_path)
    try:
        s0.reset_stage0_state()
        msg, ok = asyncio.run(s0._update_spec({"patch": {"success_criteria": [{
            "criterion_id": "c", "description": "d", "metric": "logit_diff",
            "comparator": "<=", "threshold": 0.15, "n_min": 80,
        }]}}))
        assert ok is False
        assert "logit_diff" in msg and "higher-is-better" in msg  # the real cause
        assert "required keys" not in msg  # not the old misleading framing
    finally:
        s0.reset_stage0_state()


def test_update_spec_unwraps_double_wrap_with_siblings(monkeypatch, tmp_path) -> None:
    """update_spec(patch={'patch': {...}, <sibling>}) unwraps the inner dict AND
    keeps the siblings — the 'patch contains keys patch' error is gone even when
    the double-wrap has extra fields alongside it."""
    monkeypatch.chdir(tmp_path)
    try:
        s0.reset_stage0_state()
        _, ok = asyncio.run(s0._update_spec({"patch": {
            "patch": {"question": "How is surprise represented?"},
            "hypothesis": "surprise lowers next-token confidence",
        }}))
        assert ok
        data = s0.current_partial().data
        assert data["question"].startswith("How is")
        assert data["hypothesis"].startswith("surprise lowers")
    finally:
        s0.reset_stage0_state()


def test_remove_spec_fields_rejects_phantom_patch(monkeypatch, tmp_path: Path) -> None:
    """The weak driver tries remove_spec_fields(keys=['patch']) — 'patch' is the
    update_spec argument, not a field. It must be rejected honestly (not a
    misleading '✓ removed · patch')."""
    monkeypatch.chdir(tmp_path)
    try:
        s0.reset_stage0_state()
        msg, ok = asyncio.run(s0._remove_spec_fields({"keys": ["patch"]}))
        assert ok is False
        assert "not spec fields" in msg.lower() and "patch" in msg
    finally:
        s0.reset_stage0_state()


def test_finalize_blocked_until_user_approves(monkeypatch, tmp_path: Path) -> None:
    """With the REPL's approval gate on, the agent cannot finalize (and auto-
    launch) a plan the user has not approved — the 'drafted AND ran it on turn 0
    without ever asking me' bug. The draft is kept, so the SAME plan finalizes
    the moment the user approves."""
    monkeypatch.chdir(tmp_path)
    _load(_draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    ))
    try:
        s0.set_approval_gate(True)
        s0.set_user_approved(False)  # the turn-0 question is not an approval
        msg, ok = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert ok is False
        assert "not approved" in msg.lower() and "present" in msg.lower()
        spec_dir = tmp_path / "outputs" / "specs"
        assert not (spec_dir.exists() and list(spec_dir.glob("*_rev*.json")))  # nothing ran

        # The user approves → the same kept draft finalizes and launches.
        s0.set_user_approved(True)
        msg2, ok2 = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert ok2 is True
        assert (spec_dir / "surprise-flow_rev1.json").exists()
    finally:
        s0.reset_stage0_state()


def test_finalize_gate_off_by_default(monkeypatch, tmp_path: Path) -> None:
    """Non-interactive callers (and tests) that finalize directly are unaffected:
    the gate is opt-in, so a fresh state finalizes without an approval flag."""
    monkeypatch.chdir(tmp_path)
    _load(_draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    ))
    try:
        # _load → reset_stage0_state left the gate OFF; no set_user_approved call.
        _, ok = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert ok is True
    finally:
        s0.reset_stage0_state()


def test_finalize_canonicalizes_model_alias(monkeypatch, tmp_path: Path) -> None:
    """A display/alias name (gpt2-small) must be rewritten to its real Hub repo
    (gpt2) when the spec is finalized, so the id the agent reads loads directly —
    even via a raw `from_pretrained`, which 404s on "gpt2-small". The original is
    preserved in metadata.display_name. This is the source fix for the run that
    died on `GPT2LMHeadModel.from_pretrained("gpt2-small")` → 404 → the driver
    misreading it as "hardware/network can't load the model"."""
    monkeypatch.chdir(tmp_path)
    data = _draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    )
    data["model"]["model_id"] = "gpt2-small"  # the literature/display name
    _load(data)
    try:
        msg, ok = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert ok, msg
        saved = json.loads(
            (tmp_path / "outputs" / "specs" / "surprise-flow_rev1.json").read_text()
        )
        assert saved["model"]["model_id"] == "gpt2"  # canonical Hub repo
        assert saved["model"]["metadata"]["display_name"] == "gpt2-small"
    finally:
        s0.reset_stage0_state()


def test_finalize_leaves_canonical_model_untouched(monkeypatch, tmp_path: Path) -> None:
    """A name that is already a real repo (or an unknown one) passes through with
    no display_name shim — normalization only fires for known aliases."""
    monkeypatch.chdir(tmp_path)
    data = _draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    )
    data["model"]["model_id"] = "gpt2"
    _load(data)
    try:
        _, ok = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert ok
        saved = json.loads(
            (tmp_path / "outputs" / "specs" / "surprise-flow_rev1.json").read_text()
        )
        assert saved["model"]["model_id"] == "gpt2"
        assert "display_name" not in saved["model"]["metadata"]
    finally:
        s0.reset_stage0_state()


def test_placeholder_model_blocks_validate_and_finalize(monkeypatch, tmp_path: Path) -> None:
    """A placeholder model id (the real run set 'local:/path/to/pythia-125M' from
    the prompt 'a small open-weights LOCAL model') must be caught at plan time —
    by both validate_spec and finalize_spec — so the agent picks a concrete repo
    BEFORE approval, instead of the run bailing at model-load into a confusing
    'plan change requested' panel."""
    monkeypatch.chdir(tmp_path)
    data = _draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    )
    data["model"]["model_id"] = "local:/path/to/pythia-125M"
    _load(data)
    try:
        vmsg, vok = asyncio.run(s0._validate_spec({}))
        assert vok  # tool ran; the spec is what's not ready
        assert "placeholder" in vmsg.lower() and "gpt2" in vmsg
        assert "OK" not in vmsg.split("\n")[0]

        fmsg, fok = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert fok is False
        assert "placeholder" in fmsg.lower() and "gpt2" in fmsg
        # nothing doomed was written
        spec_dir = tmp_path / "outputs" / "specs"
        assert not (spec_dir.exists() and list(spec_dir.glob("surprise-flow_rev*.json")))

        # Fixing the model to a real repo clears the block and finalizes.
        asyncio.run(s0._update_spec({"patch": {"model": "gpt2"}}))
        _, fok2 = asyncio.run(
            s0._finalize_spec({"approver": "u@x", "approver_kind": "human"})
        )
        assert fok2 is True
        assert (spec_dir / "surprise-flow_rev1.json").exists()
    finally:
        s0.reset_stage0_state()


def test_finalize_auto_bumps_revision_on_refinalize(monkeypatch, tmp_path: Path) -> None:
    """Re-finalizing after a run (e.g. the user revises the model post-failure)
    must create a NEW revision, not silently overwrite the same file — otherwise
    the REPL never sees a new spec and the investigation never launches."""
    monkeypatch.chdir(tmp_path)
    data = _draft_data(
        InvestigationStage.BLACK_BOX, PatternId.BLACKBOX_THEN_PATCHING,
        [ToolName.BLACKBOX_PROBE], [MetricName.ACCURACY],
    )
    _load(data)
    msg1, ok1 = asyncio.run(s0._finalize_spec({"approver": "u@x", "approver_kind": "human"}))
    assert ok1 and "revision 1" in msg1
    assert (tmp_path / "outputs" / "specs" / "surprise-flow_rev1.json").exists()

    # Re-finalize the same draft → auto-bump to rev2 with parent + prior set.
    _load(data)
    msg2, ok2 = asyncio.run(s0._finalize_spec({"approver": "u@x", "approver_kind": "human"}))
    assert ok2 and "revision 2" in msg2
    rev2 = tmp_path / "outputs" / "specs" / "surprise-flow_rev2.json"
    assert rev2.exists()  # a NEW path → the REPL launch trigger fires
    saved = json.loads(rev2.read_text())
    assert saved["revision"] == 2
    assert saved["parent_spec_id"] is not None
    assert saved["prior_results_ref"] is not None
    assert saved["status"] == "approved"
