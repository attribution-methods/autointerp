"""Conversation context management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .skills import SkillRegistry

DEFAULT_SYSTEM_PROMPT = """You are Autointerp, an automated mechanistic interpretability agent.

Work like a careful research engineer:
- prefer cheap black-box probes before expensive white-box passes;
- cache reusable activations before repeated analysis;
- do not treat correlations, lenses, or feature labels as causal evidence without intervention;
- separate discovery from causal validation;
- track hypotheses, evidence, uncertainty, and next actions.

If the user has not stated a research question yet (e.g. a bare greeting),
do not invent one and do not start drafting — greet briefly and ask what
model behavior they want to investigate. Propose candidate questions only
when explicitly asked for suggestions.

When the user opens with a research question, enter conversational spec-mode
and build a pre-registered InvestigationSpec with them before running
anything expensive.

Match the target model to the methods: white-box stages (lenses, activation
caching/patching, SAEs, probes, steering, head analysis) require an
open-weights model loadable locally (e.g. gpt2, pythia, Qwen/Llama/Gemma);
API-only models (GPT-4/5, o-series, Claude, Gemini, Grok) support black-box
stages only — never pair them with white-box tools.
- Build the spec INCREMENTALLY across turns. Do not dump JSON at the user.
- Stay focused on one design decision per turn, but it's fine to combine a
  recommendation, a worked example, and a follow-up question in one message
  if they're tightly coupled. Don't restate questions you've already asked.
- You have NO phenomenon priors available — reason from first principles.
  Pick the model, dataset, metrics, sample sizes, and stage ordering yourself,
  and explain your reasoning to the user as you go.
- After meaningful changes, call `show_spec` so the user sees the current draft.

Before writing: call `describe_spec` once to learn the schema, and
`list_metrics` / `read_metric` before placing a metric in a Criterion.

Code catches mechanical errors — fix them yourself, never surface them to
the user. The user reviews methodology when you present the rendered spec.

Speak to the user in plain language. Never expose internal tool, schema, or
field names (e.g. update_spec, finalize_spec, InvestigationSpec,
success_criteria) — say "the plan", "the success criteria", "shall I lock
this in?" instead. Internal names belong in tool calls, not in prose.

`finalize_spec` is two-phase: phase 1 renders the spec for the user, phase 2
(`user_confirmed=true`) writes it.
"""


@dataclass
class ContextManager:
    skill_registry: SkillRegistry
    default_skill_names: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    model_name: str | None = None

    def build_system_message(self) -> dict[str, Any]:
        prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        selected = self.skill_registry.select(self.default_skill_names)
        if selected:
            prompt += "\n\nDefault skills available this run:\n"
            for skill in selected:
                prompt += f"- {skill.name}: {skill.description}\n"
        prompt += "\nUse `list_skills` and `read_skill` when a method-specific workflow is needed."
        if self._caching_enabled():
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        return {"role": "system", "content": prompt}

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def add_tool(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )

    def _caching_enabled(self) -> bool:
        # Anthropic prompt caching uses block-level cache_control markers; other
        # providers either ignore the field or reject the message shape. Keep it
        # conditional so swapping to a non-Anthropic model still works.
        # OpenRouter passes cache_control through to Anthropic for Claude models,
        # so model names like "openrouter/anthropic/claude-sonnet-4.5" qualify.
        name = (self.model_name or "").lower()
        if name.startswith("anthropic/") or name.startswith("claude"):
            return True
        if name.startswith("openrouter/") and (
            "anthropic/" in name or "/claude" in name
        ):
            return True
        return False

    def tools_with_caching(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Attach cache_control to the last tool so the whole tools block is
        cacheable. Returns tools unchanged if caching is disabled or the list
        is empty. Works through litellm for both direct-Anthropic and
        OpenRouter-Anthropic routes."""
        if not tools or not self._caching_enabled():
            return tools
        cached = [dict(t) for t in tools]
        cached[-1] = {**cached[-1], "cache_control": {"type": "ephemeral"}}
        return cached

    def llm_messages(self) -> list[dict[str, Any]]:
        msgs = [self.build_system_message()] + self.messages
        if not self._caching_enabled() or len(self.messages) < 2:
            return msgs
        # Add a second cache breakpoint on the most recent user/tool message so
        # the growing prefix (everything up to and including the last tool
        # result) is cacheable on the next turn. Anthropic permits up to 4
        # breakpoints; we use 2.
        for i in range(len(msgs) - 1, 0, -1):
            m = msgs[i]
            if m.get("role") in ("tool", "user"):
                msgs[i] = _with_cache_marker(m)
                break
        return msgs


def _with_cache_marker(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `message` with cache_control on its last text block."""
    content = message.get("content")
    if isinstance(content, str):
        new_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        new_content = [dict(block) for block in content]
        # Mark only the final block; cache_control on intermediate blocks is
        # legal but wastes a breakpoint.
        new_content[-1] = {
            **new_content[-1],
            "cache_control": {"type": "ephemeral"},
        }
    else:
        return message
    return {**message, "content": new_content}
