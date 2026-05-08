"""Iterative circuit / feature discovery sub-agent.

This package is the autointerp port of the
``circuitbreaker/auto_circuit_discovery`` reference: an outer loop that
spawns a Claude sub-agent (via ``claude_code_sdk``) which iteratively
writes ``algorithm_v{N}.py`` candidates, evaluates them against an
autointerp-registered reward metric, and refines based on a leaderboard
that lives in a session directory.

Entry point: :func:`run_discovery_subagent`. The Tier-2 ``discover_features``
tool wraps this so the outer investigation agent can invoke it without
spawning subprocesses itself.
"""

from .candidate import COMPONENT_KINDS, FEATURE_KINDS, Candidate, FeatureCandidate  # noqa: F401
from .loop import run_discovery_subagent  # noqa: F401
from .session import update_session_files, write_leaderboard, write_memory_summary  # noqa: F401

__all__ = [
    "COMPONENT_KINDS",
    "FEATURE_KINDS",
    "Candidate",
    "FeatureCandidate",
    "run_discovery_subagent",
    "update_session_files",
    "write_leaderboard",
    "write_memory_summary",
]
