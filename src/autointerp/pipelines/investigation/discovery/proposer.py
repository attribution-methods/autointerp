"""Candidate proposers for the hill-climbing engine.

Two modes, both provider-agnostic (LiteLLM under the hood):

- ``propose_single`` — one LLM completion returning a fenced artifact block.
  Cheap; no tools. The default.
- ``propose_agentic`` — a bounded tool-using subagent via the repo's own
  ``run_agent_turn`` (+ ``ToolRouter`` with file/bash tools scoped to the
  session dir). The subagent writes the artifact file itself and may test it.

Both accumulate token/cost usage into the passed ``CostTracker`` so the engine
can bridge spend into the run budget.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_CODE_BLOCK = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)```", re.DOTALL)


def extract_artifact(text: str, *, require_token: str | None = None) -> str | None:
    """Pull the artifact body from an LLM response.

    Prefers a fenced block; falls back to raw text iff it contains
    ``require_token`` (e.g. ``"def score"``).
    """
    m = _CODE_BLOCK.search(text or "")
    if m:
        return m.group(1).strip()
    if text and (require_token is None or require_token in text):
        return text.strip()
    return None


async def propose_single(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float | None,
    cost_tracker: Any | None = None,
    require_token: str | None = None,
) -> str | None:
    """One LLM completion → artifact body (or None on failure)."""
    from autointerp_agent.agent_loop import _call_llm

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = await _call_llm(**kwargs)
    except Exception:
        return None
    if cost_tracker is not None:
        try:
            cost_tracker.add_response(resp)
        except Exception:
            pass
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        return None
    return extract_artifact(text, require_token=require_token)


async def propose_agentic(
    *,
    system_prompt: str,
    user_prompt: str,
    candidate_path: Path,
    session_dir: Path,
    model: str,
    temperature: float | None,
    max_turns: int,
    cost_tracker: Any | None = None,
) -> str | None:
    """Bounded tool-using subagent that writes the artifact itself.

    Reuses ``run_agent_turn`` so the subagent goes through the same LiteLLM +
    ToolRouter path as the master agent — no extra SDK. Returns the file
    contents written to ``candidate_path`` (or None if nothing was written).
    """
    from autointerp_agent.agent_loop import run_agent_turn
    from autointerp_agent.config import AgentConfig
    from autointerp_agent.context import ContextManager
    from autointerp_agent.skills import SkillRegistry
    from autointerp_agent.tools import ToolRouter

    try:
        registry = SkillRegistry.from_repo()
    except Exception:
        try:
            registry = SkillRegistry.from_dir("skills")
        except Exception:
            registry = SkillRegistry(skills={})

    router = ToolRouter(registry, auto_approve=True)
    # Scope file writes / bash cwd to the session dir.
    router.run_dir = session_dir
    router.writable_roots = [session_dir]
    router.scratch_dir = session_dir

    context = ContextManager(
        skill_registry=registry,
        system_prompt=system_prompt,
        model_name=model,
    )
    config = AgentConfig(
        model_name=model,
        max_iterations=max_turns,
        auto_approve=True,
        temperature=temperature,
    )
    full_prompt = (
        f"{user_prompt}\n\n"
        f"Write the full improved artifact to: {candidate_path}\n"
        f"(working directory: {session_dir}). After writing, you may run a "
        f"quick `python -c 'import ...'` style check. Finish once the file "
        f"exists and parses."
    )
    try:
        await run_agent_turn(
            full_prompt, config, context, router, cost_tracker=cost_tracker
        )
    except Exception:
        pass
    if candidate_path.exists():
        return candidate_path.read_text()
    return None


__all__ = ["extract_artifact", "propose_single", "propose_agentic"]
