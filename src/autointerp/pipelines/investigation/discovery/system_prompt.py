"""Substrate-aware system prompt builder.

Composes the discovery sub-agent's system prompt at runtime from the
shared ``_base.md`` (workflow / [DONE] / logging) + a substrate-specific
"ideas to try" block (``components.md`` or ``features.md``). Reward
metric and reward description are substituted via :class:`string.Template`
so the markdown can use ``{`` and ``}`` freely (algorithm examples,
JSON, …) without escaping.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

_PROMPTS_DIR = Path(__file__).resolve().parent / "system_prompts"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# ``DISCOVERY_SYSTEM_PROMPT_TEMPLATE`` is kept as the legacy single-
# string template for backwards-compatibility callers and tests.
DISCOVERY_SYSTEM_PROMPT_TEMPLATE = (_PROMPTS_DIR / "_base.md").read_text()


def build_discovery_system_prompt(
    *,
    reward_metric: str,
    reward_description: str,
    substrate: str = "components",
    component_kinds: list[str] | None = None,
    decomposition: str | None = None,
) -> str:
    """Compose the sub-agent system prompt.

    Parameters
    ----------
    reward_metric:
        Name of the metric being optimized.
    reward_description:
        Short paragraph describing the reward (range / direction / what
        makes it high). Pulled from ``metrics/<name>.md`` by the loop.
    substrate:
        ``"components"`` or ``"features"``. Selects which "Ideas to try"
        block is appended.
    component_kinds:
        For ``substrate="components"``: list of kinds the sub-agent may
        rank (e.g. ``["attn_head"]``).
    decomposition:
        For ``substrate="features"``: name of the decomposition family
        (e.g. ``"sae_gemmascope"``).
    """
    base = Template(DISCOVERY_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
        reward_metric=reward_metric,
        reward_description=reward_description.strip(),
    )
    if substrate == "components":
        kinds = component_kinds or ["attn_head"]
        block = Template((_PROMPTS_DIR / "components.md").read_text()).safe_substitute(
            component_kinds=", ".join(kinds),
        )
    elif substrate == "features":
        block = Template((_PROMPTS_DIR / "features.md").read_text()).safe_substitute(
            decomposition=decomposition or "<unset>",
        )
    else:
        raise ValueError(
            f"Unknown substrate {substrate!r}; expected 'components' or 'features'"
        )
    return base + "\n\n" + block


def template_path_for_substrate(substrate: str) -> Path:
    """Return the path of the algorithm template appropriate for ``substrate``.

    Used by the loop's ``init_session`` to seed the session's
    ``algorithm_template.py`` with a working baseline.
    """
    if substrate == "components":
        return _TEMPLATES_DIR / "components_baseline.py"
    if substrate == "features":
        return _TEMPLATES_DIR / "features_baseline.py"
    raise ValueError(
        f"Unknown substrate {substrate!r}; expected 'components' or 'features'"
    )


__all__ = [
    "DISCOVERY_SYSTEM_PROMPT_TEMPLATE",
    "build_discovery_system_prompt",
    "template_path_for_substrate",
]
