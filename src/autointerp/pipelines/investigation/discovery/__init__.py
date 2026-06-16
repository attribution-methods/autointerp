"""Hill-climbing discovery sub-agent.

A generic engine (``run_hillclimb``) optimizes any artifact against any reward
via iterative LLM proposals + deterministic evaluation, keeping a top-K archive
(population). ``run_discovery_subagent`` is the feature-discovery instantiation
exposed by the ``discover_features`` Tier-2 tool.

Provider-agnostic (LiteLLM); proposals run single-completion or as a tool-using
subagent (``run_agent_turn``). The sub-agent is *advisory* — it commits no gated
artifacts, so pre-registration invariants hold.
"""

from __future__ import annotations

from .candidate import Candidate
from .engine import run_hillclimb
from .loop import DiscoveryResult, run_discovery_subagent
from .task import HillClimbConfig, HillClimbResult, HillClimbTask

__all__ = [
    "Candidate",
    "DiscoveryResult",
    "HillClimbConfig",
    "HillClimbResult",
    "HillClimbTask",
    "run_discovery_subagent",
    "run_hillclimb",
]
