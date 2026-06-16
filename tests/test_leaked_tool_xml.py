"""Regression tests for the leaked-tool-xml detector in agent_loop."""

from autointerp_agent.agent_loop import _LEAKED_TOOL_XML


def test_detects_real_world_leak() -> None:
    # The literal string from logs/attn_sinks_run3.log iter 55 that killed
    # the run. The leading `<` was eaten by a stop sequence, leaving "tml:".
    leaked = (
        "tml:parameter> <parameter "
        'name="metric_result_ref">findings/stage_2_intervention/'
        "metric_c1_concentration_test.json</parameter> </invoke>"
    )
    assert _LEAKED_TOOL_XML.search(leaked) is not None


def test_detects_intact_xml_leak() -> None:
    leaked = (
        "<invoke name=\"evaluate_criterion\">"
        "<parameter name=\"criterion_id\">c1</parameter>"
        "</invoke>"
    )
    assert _LEAKED_TOOL_XML.search(leaked) is not None


def test_detects_antml_prefixed_form() -> None:
    # Construct the antml-prefixed form by concatenation so this source file
    # itself doesn't contain literal tool-call XML the harness might choke on.
    prefix = "ant" + "ml:"
    leaked = (
        f"<{prefix}invoke name=\"foo\">"
        f"<{prefix}parameter name=\"x\">1</{prefix}parameter>"
        f"</{prefix}invoke>"
    )
    assert _LEAKED_TOOL_XML.search(leaked) is not None


def test_no_false_positive_on_normal_text() -> None:
    benign_cases = [
        "Stage 2 complete; concentration ratio = 8.08 across 10 samples.",
        "I'll now call evaluate_criterion for c1.",
        "The function parameter list was reviewed.",
        "Read invoke.py to check the dispatcher.",
    ]
    for text in benign_cases:
        assert _LEAKED_TOOL_XML.search(text) is None, f"false positive on: {text!r}"


# ---------------------------------------------------------------------------
# plain-language guard (internal identifiers never reach the user)
# ---------------------------------------------------------------------------


def test_internal_jargon_pattern() -> None:
    from autointerp_agent.agent_loop import _INTERNAL_JARGON

    for leaky in (
        "Here's the current pre-registered InvestigationSpec (as drafted):",
        "Say 'approve' and I'll call finalize_spec.",
        "I used update_spec to patch the dataset.",
    ):
        assert _INTERNAL_JARGON.search(leaky), leaky
    for clean in (
        "Here's the investigation plan as drafted:",
        "Say 'approve' and I'll lock the plan in.",
        "I updated the success criteria and the dataset.",
        "### Success criteria (immutable)",  # rendered-spec output stays legal
        "  - c1: accuracy >= 0.9 on_split=heldout — behavioral sanity",
    ):
        assert not _INTERNAL_JARGON.search(clean), clean


def _fake_llm_sequence(*contents: str):
    from types import SimpleNamespace

    replies = list(contents)

    async def fake_acompletion(**kwargs):
        content = replies.pop(0) if replies else "done"
        return SimpleNamespace(
            choices=[SimpleNamespace(message={"role": "assistant", "content": content})]
        )

    return fake_acompletion


def test_jargon_guard_retries_then_passes_clean_reply(monkeypatch, tmp_path) -> None:
    import asyncio

    import autointerp_agent.agent_loop as al
    from autointerp_agent.config import AgentConfig
    from autointerp_agent.context import ContextManager
    from autointerp_agent.skills import SkillRegistry
    from autointerp_agent.tools import ToolRouter

    monkeypatch.setattr(
        al, "acompletion",
        _fake_llm_sequence(
            "Here is the InvestigationSpec for your review.",
            "Here is the investigation plan for your review.",
        ),
    )
    config = AgentConfig(plain_language_guard=True, max_iterations=5)
    registry = SkillRegistry({})
    context = ContextManager(skill_registry=registry)
    router = ToolRouter(skill_registry=registry, auto_approve=True)
    answer = asyncio.run(al.run_agent_turn("hi", config, context, router))
    assert answer == "Here is the investigation plan for your review."
    # The corrective nudge is part of the conversation record.
    assert any(
        m.get("role") == "user" and "plain language" in str(m.get("content"))
        for m in context.messages
    )


def test_jargon_guard_off_passes_through(monkeypatch) -> None:
    import asyncio

    import autointerp_agent.agent_loop as al
    from autointerp_agent.config import AgentConfig
    from autointerp_agent.context import ContextManager
    from autointerp_agent.skills import SkillRegistry
    from autointerp_agent.tools import ToolRouter

    leaky = "Here is the InvestigationSpec for your review."
    monkeypatch.setattr(al, "acompletion", _fake_llm_sequence(leaky))
    config = AgentConfig(plain_language_guard=False, max_iterations=5)
    registry = SkillRegistry({})
    context = ContextManager(skill_registry=registry)
    router = ToolRouter(skill_registry=registry, auto_approve=True)
    answer = asyncio.run(al.run_agent_turn("hi", config, context, router))
    assert answer == leaky  # pipeline contexts keep exact output


def test_jargon_retry_prompt_does_not_invite_narration() -> None:
    """The corrective nudge must NOT make the model narrate the rewrite to the
    user ("here is the same content rewritten…", "thanks for the guidance").
    It is injected as a fake user turn, so without these guards a weak model
    treats it as user guidance and thanks them for it. The prompt tells the
    model the user never saw the prior draft and to emit only the fixed reply.
    """
    from autointerp_agent.agent_loop import _JARGON_RETRY_PROMPT

    low = _JARGON_RETRY_PROMPT.lower()
    assert "not from the user" in low  # disowns the fake user turn
    assert "not seen" in low  # the user never saw the previous draft
    assert "only the corrected reply" in low  # no preface/meta
    # Forbids the exact leak the user reported.
    assert "here is the rewritten" in low
    assert "thanks for the" in low
    assert "do not mention this check" in low


# ---------------------------------------------------------------------------
# a raising tool handler must not crash the turn
# ---------------------------------------------------------------------------


def test_tool_exception_becomes_recoverable_error(monkeypatch) -> None:
    """A tool handler that raises (e.g. a malformed arg hitting an unguarded
    path → FileNotFoundError) must come back to the agent as a tool error,
    not propagate out and kill the whole turn."""
    import asyncio

    import autointerp_agent.agent_loop as al
    from autointerp_agent.agent_loop import run_agent_turn
    from autointerp_agent.config import AgentConfig
    from autointerp_agent.context import ContextManager
    from autointerp_agent.skills import SkillRegistry
    from autointerp_agent.tools import ToolRouter, ToolSpec

    class _Msg:
        def __init__(self, content="", tool_calls=None):
            self.content, self.tool_calls, self.role = content, tool_calls, "assistant"

        def model_dump(self, exclude_none=False):
            return {"role": "assistant", "content": self.content,
                    "tool_calls": self.tool_calls}

    class _Resp:
        def __init__(self, msg):
            self.choices = [type("C", (), {"message": msg})()]

    calls = {"n": 0}

    async def fake_ac(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(_Msg("run it", [
                {"id": "t1", "function": {"name": "boom", "arguments": "{}"}}]))
        return _Resp(_Msg("recovered"))

    monkeypatch.setattr(al, "acompletion", fake_ac)

    async def boom_handler(args):
        raise FileNotFoundError("")  # the exact crash shape from the report

    config = AgentConfig(model_name="fake/model", max_iterations=4)
    registry = SkillRegistry({})
    context = ContextManager(skill_registry=registry)

    async def go():
        async with ToolRouter(skill_registry=registry, auto_approve=True) as router:
            router.register_tool(ToolSpec(
                name="boom", description="d",
                parameters={"type": "object", "properties": {}}, handler=boom_handler))
            return await run_agent_turn("go", config, context, router)

    answer = asyncio.run(go())
    assert answer == "recovered"  # the turn survived the exception
    tool_msgs = [m for m in context.messages if m.get("role") == "tool"]
    assert any("FileNotFoundError" in str(m.get("content")) for m in tool_msgs)
