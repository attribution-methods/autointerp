"""C0 free-agent harness: neutral prompt, capability parity, pruned tools.

No LLM is called — only prompt/tool wiring is exercised.
"""

from __future__ import annotations

from autointerp.pipelines.investigation.main import SYSTEM_PROMPT_HEADER
from autointerp_agent.free_agent import (
    C0_TOOLS,
    NeutralContext,
    build_c0_system_prompt,
    capability_catalog,
)
from autointerp_agent.skills import SkillRegistry
from autointerp_agent.tools import ToolRouter

_METHODOLOGY_MARKERS = [
    "Inviolable rules",
    "compute_metric",
    "evaluate_criterion",
    "request_spec_revision",
    "separate discovery from causal validation",
    "do not treat correlations",
]


def test_capability_catalog_is_verbatim_and_scoped() -> None:
    cat = capability_catalog()
    # Exact substring of the scaffold header → capability parity, no drift.
    assert cat in SYSTEM_PROMPT_HEADER
    assert "autointerp.tools.model" in cat
    assert "autointerp.tools.patching" in cat
    # The scaffold-only sentence that follows the block is excluded.
    assert "current_stage" not in cat
    assert "Inviolable" not in cat


def test_c0_prompt_is_neutral_but_keeps_capability() -> None:
    q = "How does GPT-2 do the thing? (BLINDED-FIXTURE-QUESTION)"
    prompt = build_c0_system_prompt(q, run_dir="/tmp/eval/c0-x")
    assert "BLINDED-FIXTURE-QUESTION" in prompt
    assert "autointerp.tools.model" in prompt  # capability kept
    assert "/tmp/eval/c0-x" in prompt
    for marker in _METHODOLOGY_MARKERS:
        assert marker not in prompt, f"C0 prompt leaked scaffold marker: {marker!r}"


def test_router_pruned_to_neutral_toolset() -> None:
    router = ToolRouter(skill_registry=SkillRegistry({}), auto_approve=True)
    # Defaults auto-register Stage-0 + skill tools; the harness prunes them.
    assert "describe_spec" in router.tools  # present before prune
    router.tools = {k: v for k, v in router.tools.items() if k in C0_TOOLS}
    assert set(router.tools) == set(C0_TOOLS)
    for gone in ("describe_spec", "finalize_spec", "update_spec",
                 "list_skills", "read_skill"):
        assert gone not in router.tools


def test_neutral_context_strips_skill_nudge_keeps_caching() -> None:
    sp = "NEUTRAL-PROMPT-BODY"
    ctx = NeutralContext(
        skill_registry=SkillRegistry({}),
        default_skill_names=[],
        system_prompt=sp,
        model_name="anthropic/claude-sonnet-4-5",
    )
    msg = ctx.build_system_message()
    # Anthropic model → cached block, exact body, no scaffold additions.
    assert msg["role"] == "system"
    assert msg["content"][0]["text"] == sp
    assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Default skills available" not in sp
    assert "list_skills" not in msg["content"][0]["text"]

    ctx_plain = NeutralContext(
        skill_registry=SkillRegistry({}),
        default_skill_names=[],
        system_prompt=sp,
        model_name="gpt-4o",
    )
    msg2 = ctx_plain.build_system_message()
    assert msg2 == {"role": "system", "content": sp}
