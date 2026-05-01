"""Conversation context management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .skills import SkillRegistry


DEFAULT_SYSTEM_PROMPT = """You are Autointerp, an automated mechanistic interpretability agent.

Work like a careful research engineer:
- prefer cheap black-box probes before expensive white-box passes;
- cache reusable activations before repeated analysis;
- use skills to choose methods and avoid known failure modes;
- do not treat correlations, lenses, or feature labels as causal evidence without intervention;
- maintain a concise plan when work has multiple steps;
- ask for approval before destructive local commands or expensive jobs.
"""


@dataclass
class ContextManager:
    skill_registry: SkillRegistry
    default_skill_names: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)

    def build_system_message(self) -> dict[str, str]:
        prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        selected = self.skill_registry.select(self.default_skill_names)
        if selected:
            prompt += "\n\nDefault skills available this run:\n"
            for skill in selected:
                prompt += f"- {skill.name}: {skill.description}\n"
        prompt += "\nUse `list_skills` and `read_skill` when a method-specific workflow is needed."
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

    def llm_messages(self) -> list[dict[str, Any]]:
        return [self.build_system_message()] + self.messages
