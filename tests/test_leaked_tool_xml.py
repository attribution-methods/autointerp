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
