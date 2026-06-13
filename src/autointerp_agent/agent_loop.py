"""LiteLLM-based agent loop with tool calling."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .config import AgentConfig
from .context import ContextManager
from .tools import ToolRouter

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

# Internal identifiers that must never reach the user's eyes: schema class
# names and the Stage-0 tool surface. The concepts are fine — the user hears
# "the investigation plan" and "I'll lock it in" — but the code vocabulary
# means nothing to them. Enforced (when config.plain_language_guard is on)
# with the same detect-and-retry pattern as the XML-leak guard above.
# Deliberately narrow: rendered-spec output and plain-English phrases like
# "success criteria" never match.
_INTERNAL_JARGON = re.compile(
    r"\b(InvestigationSpec|PartialSpec|StageSpec|BehaviorSpec"
    r"|finalize_spec|update_spec|describe_spec|show_spec|validate_spec"
    r"|remove_spec_fields|list_metrics|read_metric|propose_custom_metric)\b"
)
_MAX_JARGON_RETRIES = 2
_JARGON_RETRY_PROMPT = (
    "[automated style check — this is NOT from the user, and the user has NOT "
    "seen your previous reply] That reply used internal code identifiers (e.g. "
    "InvestigationSpec, finalize_spec) that mean nothing to the user. Send the "
    "reply again with identical substance but in plain language: 'the "
    "investigation plan', and actions as actions ('I'll lock the plan in', "
    "'I updated the success criteria'). Output ONLY the corrected reply — no "
    "preface, no 'here is the rewritten version', no 'thanks for the "
    "guidance', and do NOT mention this check or that anything was reworded. "
    "To the user, what you write now is your first and only reply."
)

# Resolved lazily on the first agent turn: `import litellm` reads thousands
# of module files (~12s on network filesystems) and must never delay CLI
# startup or the banner. Tests may pre-assign a fake here — any non-None
# value is used as-is.
acompletion = None

_warm_thread: threading.Thread | None = None


def _import_llm_runtime() -> Any:
    """Blocking: import litellm and pre-load its token encodings.

    Runs in a worker thread (never on the event loop — it would freeze every
    concurrent UI task, e.g. the elapsed-time ticker, for the duration).
    """
    from litellm import acompletion as _acompletion

    try:
        import litellm

        # Errors are rendered by format_llm_error — keep litellm's
        # "Give Feedback / Get Help" banners out of the shell.
        litellm.suppress_debug_info = True
        # First token count lazily loads tiktoken encodings (can hit the
        # network/disk for seconds) — pay it here, off the hot path.
        litellm.token_counter(model="gpt-4o", text="warm")
    except Exception:  # noqa: BLE001 — warming is best-effort
        pass
    return _acompletion


def warm_llm_runtime() -> None:
    """Begin importing litellm in a daemon thread (idempotent).

    Called when an interactive session opens: the user spends longer reading
    the banner and typing than the import takes, so the first turn starts
    warm instead of stalling.
    """
    global _warm_thread
    if acompletion is not None:
        return
    if _warm_thread is not None and _warm_thread.is_alive():
        return

    def _job() -> None:
        global acompletion
        try:
            result = _import_llm_runtime()
            if acompletion is None:
                acompletion = result
        except Exception:  # noqa: BLE001 — the first _call_llm will surface it
            pass

    _warm_thread = threading.Thread(target=_job, daemon=True, name="litellm-warm")
    _warm_thread.start()


async def _call_llm(**kwargs: Any) -> Any:
    global acompletion
    if acompletion is None:
        # Worker thread, not the loop: if the background warm-up is already
        # mid-import this just blocks on the import lock over there.
        acompletion = await asyncio.to_thread(_import_llm_runtime)
    return await acompletion(**kwargs)


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
    jargon_retries = 0
    # Resolve the temperature once: None (or a reasoning model that rejects a
    # custom value) means we omit it and let the provider default apply.
    temperature = getattr(config, "temperature", None)
    if temperature is not None:
        from .model_select import model_supports_temperature

        if not model_supports_temperature(config.model_name):
            temperature = None
    for iteration in range(config.max_iterations):
        await _emit(observer, "on_iteration_start", iteration)
        llm_kwargs: dict[str, Any] = {
            "model": config.model_name,
            "messages": context.llm_messages(),
            "tools": context.tools_with_caching(tool_router.get_tool_specs_for_llm()),
            "tool_choice": "auto",
            "stream": False,
        }
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        response = await _call_llm(**llm_kwargs)
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
            if (
                getattr(config, "plain_language_guard", False)
                and _INTERNAL_JARGON.search(content)
                and jargon_retries < _MAX_JARGON_RETRIES
            ):
                jargon_retries += 1
                context.add_assistant({"role": "assistant", "content": content})
                context.add_user(_JARGON_RETRY_PROMPT)
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
                try:
                    output, ok = await tool_router.call_tool(name, args)
                except Exception as exc:  # noqa: BLE001
                    # A tool handler that raises (e.g. a malformed arg hitting
                    # an unguarded path) must not crash the whole turn — return
                    # it to the agent as a recoverable tool error instead.
                    output, ok = (
                        f"Tool {name!r} raised {type(exc).__name__}: {exc}", False
                    )
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
