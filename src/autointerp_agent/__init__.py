"""Agent runtime for automated interpretability work.

Submodule attributes are exported lazily (PEP 562) so that importing the
package costs nothing: the ``autointerp`` console script must be able to
catch a Ctrl-C during its heavy imports (pydantic, litellm, …), which it can
only do if ``import autointerp_agent`` itself is instant.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["AgentConfig", "Skill", "SkillRegistry", "load_config"]

if TYPE_CHECKING:  # static analyzers see the real symbols
    from .config import AgentConfig, load_config
    from .skills import Skill, SkillRegistry

_LAZY = {
    "AgentConfig": ("autointerp_agent.config", "AgentConfig"),
    "load_config": ("autointerp_agent.config", "load_config"),
    "Skill": ("autointerp_agent.skills", "Skill"),
    "SkillRegistry": ("autointerp_agent.skills", "SkillRegistry"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value  # cache for subsequent lookups
    return value
