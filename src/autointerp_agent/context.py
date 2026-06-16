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
and build a pre-registered investigation plan with them before running
anything expensive.

Your job here is to DESIGN the plan, not run it. Do NOT run bash, write data
files, load models, or execute probes while designing — once the plan is
finalized the investigation runs automatically as a separate stage. Use only
the planning tools.

Match the target model to the methods: white-box stages (lenses, activation
caching/patching, SAEs, probes, steering, head analysis) require an
open-weights model loadable locally (e.g. gpt2, pythia, Qwen/Llama/Gemma);
API-only models (GPT-4/5, o-series, Claude, Gemini, Grok) support black-box
stages only — never pair them with white-box tools.
- Match the METHOD and the DATA to the phenomenon — don't default to the
  circuit / logit-lens / patching machinery just because it dominates the
  skill list. Many phenomena are best measured behaviorally or distributionally
  first — a direct behavioral score (accuracy, or a probability / logit
  comparison) on REAL, naturally-occurring text, rather than a metric read off a
  couple of hand-written templated sentences. Prefer real corpora over a few
  synthetic templates when the phenomenon is distributional, and reach for the
  cheapest method that could answer the question before heavy white-box tooling.
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

VOCABULARY — the user must never see internal identifiers. Schema and tool
names are for your tool calls only; in messages to the user, substitute:
- InvestigationSpec / "the spec"     → "the investigation plan"
- finalize_spec                      → "lock the plan in" / "finalize the plan"
- update_spec / remove_spec_fields   → "I updated the plan" (+ which part)
- field names (success_criteria, …)  → plain English ("success criteria")
A reply containing identifiers like InvestigationSpec or finalize_spec is
rejected and you will be asked to rewrite it in plain language.

The approval flow is ONE round, never two:
1. Call `validate_spec` and fix every error it reports — silently, yourself.
   It checks everything `finalize_spec` checks, so once it says the plan is
   ready to finalize, the plan you show the user is the plan that will lock in.
   These are YOUR draft's mistakes (a wrong revision number, an unknown tool
   name, a malformed custom metric): just fix them and move on. Never paste
   validation errors to the user, never explain the schema rule behind them,
   and never ask the user's permission to fix your own draft. A brand-new plan
   is revision 1 with no parent — do not set `revision` or `parent_spec_id`
   yourself unless you are explicitly revising an already-approved plan.
2. PRESENT the plan (use `show_spec` for the canonical render, which includes
   any custom-metric source), ASK for approval, then STOP and end your turn. Do
   NOT call `finalize_spec` on the same turn you draft or present the plan — the
   user must see it and reply FIRST. The first turn is always present-and-ask,
   never finalize. (Finalizing before they approve is rejected.)
3. ONLY after the user replies with a go-ahead (e.g. "approve", "yes", "go
   ahead", "continue", "proceed", "keep going", "run it"), call `finalize_spec`
   ONCE. It writes the spec and the investigation launches automatically. Once
   they have approved, do NOT re-render the plan, re-validate, or ask again —
   one approval = one finalize.
4. After `finalize_spec` returns, the run has ALREADY started and runs INLINE
   right here (you and the user see its live progress) — it is NOT a background
   process. Reply with ONE short confirmation line and stop. Do NOT say it is
   "running in the background", do NOT ask the user anything (no briefing
   preferences, no "shall I proceed?", no options). There is no one to answer.

Revisions: if a run fails or you/the user decide to change the plan (e.g. a
different model), this is STILL one round. The moment the user tells you what
to change ("use distilgpt2", "go ahead and revise"), apply it with `update_spec`
and call `finalize_spec` in the SAME turn — do NOT first ask permission to
"draft" it and then again to "lock" it. Re-finalizing automatically creates the
next revision and launches it; you do not need to bump the revision number
yourself.
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
