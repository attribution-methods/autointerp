"""LiteLLM-based agent loop with tool calling."""

from __future__ import annotations

import json
from typing import Any

from litellm import acompletion

from .config import AgentConfig
from .context import ContextManager
from .tools import ToolRouter


async def run_agent_turn(
    user_prompt: str,
    config: AgentConfig,
    context: ContextManager,
    tool_router: ToolRouter,
) -> str:
    context.add_user(user_prompt)
    final_text = ""
    for _iteration in range(config.max_iterations):
        response = await acompletion(
            model=config.model_name,
            messages=context.llm_messages(),
            tools=tool_router.get_tool_specs_for_llm(),
            tool_choice="auto",
            stream=False,
        )
        message = response.choices[0].message
        assistant_message = _message_to_dict(message)
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            final_text = str(assistant_message.get("content") or "")
            context.add_assistant({"role": "assistant", "content": final_text})
            return final_text
        context.add_assistant(assistant_message)
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name")
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                output, ok = f"Malformed JSON arguments: {exc}", False
            else:
                output, ok = await tool_router.call_tool(name, args)
            prefix = "" if ok else "ERROR: "
            context.add_tool(tool_call.get("id", name), name, prefix + output)
    return "Stopped after max_iterations without a final answer."


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", None),
    }
