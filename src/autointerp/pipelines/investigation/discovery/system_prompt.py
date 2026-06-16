"""System prompt for the hill-climbing sub-agent's proposal step.

Single-string and task-driven — the artifact contract is injected per task
(``HillClimbTask.contract_instructions``) so the same prompt scaffold serves
feature discovery, prompt optimization, probe search, etc. Reward metric and
description are substituted via :class:`string.Template` so the body can use
``{`` / ``}`` freely in code examples without escaping.
"""

from __future__ import annotations

from string import Template

_SYSTEM_PROMPT = """\
You are a hill-climbing search agent optimizing an artifact for a target task.
Each turn you propose ONE improved candidate that maximizes the reward metric
`$reward_metric` (higher is better). $reward_description

## Artifact contract
$contract_instructions

## How the search works
You are hill-climbing with a small population (archive) of the best candidates
so far. Each turn you are given a parent to improve, the current archive (best
candidates + rewards), and the recent trail (what was tried and how it scored,
including failures). Make ONE focused, well-motivated change you believe will
raise the reward — reuse what worked, avoid repeating what failed. Do not
rewrite from scratch unless the parent is the trivial baseline.

$response_instructions
"""

# Single-completion mode: emit exactly one fenced code/artifact block.
_RESPONSE_SINGLE = (
    "Respond with EXACTLY ONE fenced code block containing the full artifact "
    "(imports + everything required by the contract). No prose outside the "
    "block."
)

# Agentic mode: the subagent has file/bash tools and writes the artifact itself.
_RESPONSE_AGENTIC = (
    "You have file and bash tools. Write the full improved artifact to the "
    "path given in the task prompt, then (optionally) run a quick check that it "
    "imports / parses. Finish once the file is written."
)


def build_hillclimb_system_prompt(
    *,
    reward_metric: str,
    reward_description: str,
    contract_instructions: str,
    agentic: bool = False,
) -> str:
    """Compose the sub-agent system prompt for a task + propose mode."""
    return Template(_SYSTEM_PROMPT).safe_substitute(
        reward_metric=reward_metric,
        reward_description=(reward_description or "").strip(),
        contract_instructions=contract_instructions.strip(),
        response_instructions=_RESPONSE_AGENTIC if agentic else _RESPONSE_SINGLE,
    )


# Backwards-compatible alias for the feature-discovery contract.
_DISCOVERY_CONTRACT = (
    "Define a top-level `def score(pairs, *, loader, layers, device, top_k, "
    "context=None)` that ranks model components/features and returns a "
    "`list[Candidate]` (from "
    "`autointerp.pipelines.investigation.discovery.candidate`), sorted by "
    "descending `abs(score)`, length <= top_k. Pure ranking only: no I/O, no "
    "global mutation; the harness owns model loading, ablation, steering, and "
    "metric computation."
)


def build_discovery_system_prompt(
    *, reward_metric: str, reward_description: str = ""
) -> str:
    """Feature-discovery system prompt (single-completion mode)."""
    return build_hillclimb_system_prompt(
        reward_metric=reward_metric,
        reward_description=reward_description,
        contract_instructions=_DISCOVERY_CONTRACT,
        agentic=False,
    )


__all__ = [
    "build_hillclimb_system_prompt",
    "build_discovery_system_prompt",
]
