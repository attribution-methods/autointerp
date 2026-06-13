"""Agent sampling-temperature control: the reasoning-model guard, the
AUTOINTERP_TEMPERATURE env override, and the agent-loop pass-through (send it
when set + supported, drop it for reasoning models, omit it when unset)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from autointerp_agent import agent_loop as al
from autointerp_agent.config import AgentConfig, env_default_temperature
from autointerp_agent.context import ContextManager
from autointerp_agent.model_select import model_supports_temperature
from autointerp_agent.skills import SkillRegistry
from autointerp_agent.tools import ToolRouter


def test_model_supports_temperature() -> None:
    # OpenAI reasoning models reject a custom temperature.
    for reasoning in (
        "openai/gpt-5-nano", "gpt-5-nano", "openai/gpt-5.1", "openai/o1",
        "o3-mini", "openai/o4-mini", "openrouter/openai/o1-preview",
    ):
        assert model_supports_temperature(reasoning) is False, reasoning
    # Everything else honors it.
    for standard in (
        "openai/gpt-4o", "gpt-4o-mini", "anthropic/claude-sonnet-4-5",
        "hosted_vllm/Qwen/Qwen2.5-7B-Instruct", "openrouter/meta-llama/llama-3.1-70b",
    ):
        assert model_supports_temperature(standard) is True, standard


def test_env_default_temperature(monkeypatch) -> None:
    monkeypatch.delenv("AUTOINTERP_TEMPERATURE", raising=False)
    assert env_default_temperature() is None
    monkeypatch.setenv("AUTOINTERP_TEMPERATURE", "0.7")
    assert env_default_temperature() == 0.7
    monkeypatch.setenv("AUTOINTERP_TEMPERATURE", "garbage")  # never crashes startup
    assert env_default_temperature() is None


def _capture_acompletion(captured: dict):
    async def fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message={"role": "assistant", "content": "done"})]
        )

    return fake


def _run_turn(config: AgentConfig, captured: dict, monkeypatch) -> None:
    monkeypatch.setattr(al, "acompletion", _capture_acompletion(captured))
    registry = SkillRegistry({})
    context = ContextManager(skill_registry=registry)
    router = ToolRouter(skill_registry=registry, auto_approve=True)
    asyncio.run(al.run_agent_turn("hi", config, context, router))


def test_agent_turn_sends_temperature_when_supported(monkeypatch) -> None:
    captured: dict = {}
    _run_turn(
        AgentConfig(model_name="anthropic/claude-sonnet-4-5",
                    temperature=0.3, max_iterations=2),
        captured, monkeypatch,
    )
    assert captured.get("temperature") == 0.3


def test_agent_turn_drops_temperature_for_reasoning_model(monkeypatch) -> None:
    captured: dict = {}
    _run_turn(
        AgentConfig(model_name="openai/gpt-5-nano",
                    temperature=0.3, max_iterations=2),
        captured, monkeypatch,
    )
    assert "temperature" not in captured  # would 400 the provider otherwise


def test_agent_turn_omits_temperature_when_unset(monkeypatch) -> None:
    captured: dict = {}
    _run_turn(
        AgentConfig(model_name="anthropic/claude-sonnet-4-5", max_iterations=2),
        captured, monkeypatch,
    )
    assert "temperature" not in captured  # provider default applies
