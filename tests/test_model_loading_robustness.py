"""Regression tests for the class of failure where the agentic loop "died"
trying to load a model — and the driver misread the failure as a hardware /
network limit and bailed to a spec revision.

Two real root causes are covered, plus a real end-to-end smoke test:

* Bug A — the agent ran `from_pretrained("gpt2-small")` (a display alias, not a
  Hub repo) → 404. Fixed at the source by canonicalizing the spec model id at
  finalize (see test_finalize_flow) and by a torch-free alias module here.
* Bug B — the bash hard timeout (120s) was *below* the floor cost of one model
  load on this box (~130s: import torch+transformers + CUDA init), so every
  model-loading bash was killed mid-load. Fixed by raising DEFAULT_TIMEOUT.
* A backstop: `request_spec_revision` now refuses an environment-blaming reason
  when no model was ever actually loaded (no model_forward capture exists).

The smoke test (`test_real_model_load_capture_smoke`) is the thing the mocked
unit tests never did: actually load a model and run it. It is gated behind
AUTOINTERP_RUN_MODEL_SMOKE=1 so the normal suite stays fast, but it is a real,
runnable proof of the load → forward → capture path on any box with torch.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autointerp.pipelines.investigation import init_run
from autointerp.pipelines.investigation.tools import (
    _blames_environment,
    _has_model_forward_capture,
    create_investigation_tools,
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
from autointerp.tools import provenance as P
from autointerp_agent.tools import DEFAULT_TIMEOUT


def _spec() -> InvestigationSpec:
    return InvestigationSpec(
        spec_id="load-robustness",
        revision=1,
        question="Does the model X?",
        hypothesis="L9H9 mediates X.",
        phenomenon_id="custom",
        behavior=BehaviorSpec(behavior_id="b", description="d"),
        model=ModelRef(model_id="gpt2"),
        dataset=DatasetSpec(
            dataset_id="ds", source="generated", n_samples=10, split="dev", seed=1
        ),
        contrast=ContrastSpec(
            contrast_id="c", positive_template="{x}", negative_template="{y}",
            pairing="matched",
        ),
        stages=[StageSpec(
            stage=InvestigationStage.BLACK_BOX,
            pattern=PatternId.BLACKBOX_THEN_PATCHING,
            tools=[ToolName.BLACKBOX_PROBE],
            metrics=[MetricName.ACCURACY],
        )],
        success_criteria=[Criterion(
            criterion_id="c1", description="dev acc", metric=MetricName.ACCURACY,
            comparator=">=", threshold=0.9, on_split="dev",
        )],
        budget=Budget(),
        status=SpecStatus.APPROVED,
        approval=Approval(
            approver="t", approver_kind="agent",
            approved_at=datetime.now(timezone.utc),
        ),
    )


def _revision_handler(handle):
    tools = create_investigation_tools(handle)
    spec = next(t for t in tools if t.name == "request_spec_revision")
    return spec.handler


# ---------------------------------------------------------------------------
# Bug B: the hard timeout must sit above the floor cost of a model load
# ---------------------------------------------------------------------------


def test_default_timeout_exceeds_model_load_floor() -> None:
    """import torch+transformers + CUDA init + load is ~130s on this box; the
    default bash timeout must comfortably exceed it or every model-loading bash
    is killed mid-load. Guards against anyone restoring the old 120s."""
    assert DEFAULT_TIMEOUT >= 600, (
        f"DEFAULT_TIMEOUT={DEFAULT_TIMEOUT}s is below a model-load floor; a "
        "model-loading bash would be killed and misread as 'can't load the model'"
    )


# ---------------------------------------------------------------------------
# Bug A: the alias module is torch-free so finalize/validate can canonicalize
# ---------------------------------------------------------------------------


def test_model_aliases_module_is_torch_free() -> None:
    """Importing the alias module (used by spec finalize) must NOT pull torch —
    otherwise the lazy `autointerp.tools.__init__` regressed and every spec
    operation pays a ~10s torch import (and hard-depends on a GPU stack)."""
    code = (
        "import sys, autointerp.tools.model_aliases as m;"
        "assert 'torch' not in sys.modules, 'torch leaked into the alias path';"
        "assert m.resolve_model_id('gpt2-small') == 'gpt2';"
        "assert m.resolve_model_id('pythia-70m') == 'EleutherAI/pythia-70m';"
        "assert m.resolve_model_id('org/brand-new-2026') == 'org/brand-new-2026'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Backstop: refuse an environment-blaming revision when nothing ever loaded
# ---------------------------------------------------------------------------


def test_blames_environment_matches_env_excuses_only() -> None:
    # The exact reason the dead run emitted.
    assert _blames_environment(
        "Hardware/network restrictions prevent loading a real model (gpt2-small)."
    )
    assert _blames_environment("switch to offline mode / provide a pre-warmed capture")
    assert _blames_environment("cannot load the model in this environment")
    # A genuine methodological revision must pass through untouched.
    assert not _blames_environment(
        "results contradict the hypothesis: L10H7, not L9H9, carries the signal"
    )
    assert not _blames_environment("the dev split is too small to support a PASS")


def test_has_model_forward_capture_reflects_captures_dir(tmp_path: Path) -> None:
    handle = init_run(_spec(), runs_root=tmp_path)
    assert _has_model_forward_capture(handle) is False  # nothing run yet
    P.record_to(handle.root, "logit_inputs", {"target_logits": [1.0, 2.0]},
                source="model_forward")
    assert _has_model_forward_capture(handle) is True


def test_revision_rejected_for_env_excuse_without_capture(tmp_path: Path) -> None:
    """The dead-run shape: the agent wrote+ran a script (so it 'did work') but
    never actually loaded the model, then blamed hardware. That revision must be
    bounced back to fixing the script — not accepted."""
    handle = init_run(_spec(), runs_root=tmp_path)
    # Make _did_any_empirical_work True (a script was written) but NO capture.
    (handle.root / "scripts").mkdir(exist_ok=True)
    (handle.root / "scripts" / "run.py").write_text("print('boom')\n")
    handler = _revision_handler(handle)
    msg, ok = asyncio.run(handler({
        "reason": "Hardware/network restrictions prevent loading a real model gpt2-small",
    }))
    assert ok is False
    assert "load_model" in msg  # points at the real fix
    assert "from_pretrained" in msg


def test_revision_allowed_after_a_real_capture(tmp_path: Path) -> None:
    """Once the model has actually run (a capture exists), an env-flavored reason
    is no longer auto-bounced — the agent has earned the revision."""
    handle = init_run(_spec(), runs_root=tmp_path)
    P.record_to(handle.root, "inp", {"target_logits": [1.0]}, source="model_forward")
    handler = _revision_handler(handle)
    msg, ok = asyncio.run(handler({
        "reason": "after loading, the network is genuinely too small for the effect",
    }))
    assert ok is True, msg


# ---------------------------------------------------------------------------
# The real end-to-end smoke test (opt-in): load → forward → capture
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("AUTOINTERP_RUN_MODEL_SMOKE"),
    reason="set AUTOINTERP_RUN_MODEL_SMOKE=1 to run the real model load+forward",
)
def test_real_model_load_capture_smoke(tmp_path: Path) -> None:
    """Actually exercise the path the loop kept dying on: load the model BY ITS
    SPEC NAME via load_model (which resolves gpt2-small -> gpt2), run a real
    forward pass, and record the outputs as a model_forward capture."""
    import torch

    from autointerp.tools.model import load_model, resolve_model_id

    assert resolve_model_id("gpt2-small") == "gpt2"
    h = load_model("gpt2-small")  # the exact call that 404'd via raw transformers
    ids = h.tokenizer("The capital of France is", return_tensors="pt").input_ids
    ids = ids.to(h.device)
    with torch.no_grad():
        logits = h.model(ids).logits
    assert logits.shape[0] == 1 and logits.shape[-1] > 100

    handle = init_run(_spec(), runs_root=tmp_path)
    rel = P.record_to(
        handle.root, "next_token_logits",
        {"logits": logits[0, -1, :8].float().tolist()},
        source="model_forward", model_id="gpt2",
    )
    assert (handle.root / rel).exists()
    assert _has_model_forward_capture(handle) is True
