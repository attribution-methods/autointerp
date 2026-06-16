"""Minimal hill-climbing discovery sub-agent.

The ``discover_features`` Tier-2 tool delegates an iterative search to this
package: propose a candidate ranking algorithm, evaluate it against a reward
metric, keep the best, and repeat (hill-climbing). It is deliberately small —
no leaderboard / memory files, no substrate registry, no extra LLM SDK. The
proposal step reuses the same LiteLLM path as the rest of the agent runtime
(``autointerp_agent.agent_loop``), and the reward reuses the canonical metric
registry (``combined_auc_k``).

The sub-agent is *advisory*: it never commits gated artifacts. The master
agent records evidence via the normal ``commit_artifact`` / ``compute_metric``
gates, so all pre-registration invariants hold.
"""

from __future__ import annotations

from .candidate import Candidate
from .loop import DiscoveryResult, run_discovery_subagent

__all__ = ["Candidate", "DiscoveryResult", "run_discovery_subagent"]
