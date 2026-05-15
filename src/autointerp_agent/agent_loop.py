"""LiteLLM-based agent loop with tool calling."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

# Pattern for tool-call XML syntax leaking into assistant text content.
# Anthropic's models occasionally regress to the legacy <invoke>/<parameter>
# XML format inside content blocks instead of using structured tool_use blocks.
# When that happens the harness sees text-only content and would otherwise
# treat the turn as the agent's final answer; we detect and retry instead.
# The leading `<` is sometimes truncated by stop sequences (e.g. "tml:"
# instead of "<" or "<param" instead of "<parameter"), so the pattern
# matches both intact and truncated leaks.
_LEAKED_TOOL_XML = re.compile(
    r"</?(?:antml:)?(?:invoke|parameter|tool_use)\b|\btml:(?:invoke|parameter)\b",
    re.IGNORECASE,
)
_MAX_LEAKED_XML_RETRIES = 3

from litellm import acompletion

from .config import AgentConfig
from .context import ContextManager
from .tools import ToolRouter


@runtime_checkable
class TurnObserver(Protocol):
    """Hook surface for transcript capture and live console streaming.

    Each method may be sync or async; the loop awaits if needed. All methods
    are optional — callers can subclass and override only what they care
    about.
    """

    def on_iteration_start(self, iteration: int) -> Any: ...
    def on_assistant(self, iteration: int, message: dict[str, Any]) -> Any: ...
    def on_tool_call(
        self,
        iteration: int,
        call_idx: int,
        call_id: str,
        name: str,
        args: dict[str, Any],
        output: str,
        ok: bool,
    ) -> Any: ...
    def on_final(self, iteration: int, final_text: str) -> Any: ...


async def _maybe_await(maybe_coro: Any) -> None:
    if inspect.isawaitable(maybe_coro):
        await maybe_coro


async def _emit(observer: TurnObserver | None, method_name: str, *args: Any) -> None:
    if observer is None:
        return
    fn: Callable[..., Awaitable[Any] | Any] | None = getattr(observer, method_name, None)
    if fn is None:
        return
    try:
        await _maybe_await(fn(*args))
    except Exception:
        # Observers must never break the loop — a bad logger is not a bug we
        # propagate to the agent.
        pass


async def run_agent_turn(
    user_prompt: str,
    config: AgentConfig,
    context: ContextManager,
    tool_router: ToolRouter,
    observer: TurnObserver | None = None,
    cost_tracker: Any | None = None,
) -> str:
    context.add_user(user_prompt)
    final_text = ""
    leaked_xml_retries = 0
    for iteration in range(config.max_iterations):
        await _emit(observer, "on_iteration_start", iteration)
        response = await acompletion(
            model=config.model_name,
            messages=context.llm_messages(),
            tools=context.tools_with_caching(tool_router.get_tool_specs_for_llm()),
            tool_choice="auto",
            stream=False,
        )
        if cost_tracker is not None:
            try:
                cost_tracker.add_response(response)
            except Exception:
                pass
        message = response.choices[0].message
        assistant_message = _message_to_dict(message)
        await _emit(observer, "on_assistant", iteration, assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            content = str(assistant_message.get("content") or "")
            # Catch the failure mode where the model emits tool-call XML in
            # text content instead of returning a structured tool_use block.
            # Without this, the harness exits believing the agent is done.
            if (
                _LEAKED_TOOL_XML.search(content)
                and leaked_xml_retries < _MAX_LEAKED_XML_RETRIES
            ):
                leaked_xml_retries += 1
                context.add_assistant({"role": "assistant", "content": content})
                context.add_user(
                    "Your last response contained tool-call XML syntax "
                    "(e.g. <invoke>, <parameter>) inside the text content "
                    "instead of a structured tool_use block. That format is "
                    "not supported here. Re-issue the intended tool call "
                    "using the structured tool-calling API; do not include "
                    "tool XML in text content."
                )
                continue
            final_text = content
            context.add_assistant({"role": "assistant", "content": final_text})
            await _emit(observer, "on_final", iteration, final_text)
            return final_text
        leaked_xml_retries = 0
        context.add_assistant(assistant_message)
        for call_idx, tool_call in enumerate(tool_calls):
            function = tool_call.get("function", {})
            name = function.get("name")
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                output, ok = f"Malformed JSON arguments: {exc}", False
                args = {"_raw": raw_args}
            else:
                output, ok = await tool_router.call_tool(name, args)
            prefix = "" if ok else "ERROR: "
            call_id = tool_call.get("id", name)
            context.add_tool(call_id, name, prefix + output)
            await _emit(
                observer,
                "on_tool_call",
                iteration,
                call_idx,
                call_id,
                name,
                args,
                output,
                ok,
            )
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
