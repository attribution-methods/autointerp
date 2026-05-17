"""Registry-vs-enforcement contract consistency.

Root cause of the scaffold-eval `full` budget blow-up: the advertised metric
contract (`spec.METRIC_META[*].requires_inputs`, surfaced by the `read_metric`
tool) diverged from what `compute_and_commit_metric` actually enforces
(`metrics.py` `_require_keys(...)`). The agent could not comply by reading the
contract because the contract was wrong; it had to fail-and-learn.

This test pins the invariant: for every metric that is commit-validated, the
*advertised* required inputs must equal the *enforced* required inputs. It
parses the literal `_require_keys(MetricName.X, inputs, (...))` calls — it does
not execute metric functions — so enforcement remains the single source of
truth and the registry must track it.
"""
from __future__ import annotations

import re
from pathlib import Path

from autointerp.spec import METRIC_META

_METRICS_PY = (
    Path(__file__).resolve().parents[1]
    / "src/autointerp/pipelines/investigation/metrics.py"
)


def _enforced_required_keys() -> dict[str, list[str]]:
    src = _METRICS_PY.read_text()
    out: dict[str, list[str]] = {}
    for name, keys in re.findall(
        r"_require_keys\(\s*MetricName\.(\w+)\s*,\s*inputs\s*,\s*\(([^)]*)\)",
        src,
        re.S,
    ):
        out[name.lower()] = [
            k.strip().strip('"').strip("'") for k in keys.split(",") if k.strip()
        ]
    return out


def test_registry_requires_inputs_matches_enforcement() -> None:
    enforced = _enforced_required_keys()
    assert enforced, "no _require_keys(...) calls parsed — parser/anchor stale"
    mismatches = []
    for mn, meta in METRIC_META.items():
        keys = enforced.get(mn.value)
        if keys is None:
            continue  # metric is not commit-validated; no contract to pin
        if sorted(meta.requires_inputs) != sorted(keys):
            mismatches.append(
                f"{mn.value}: registry {meta.requires_inputs} != enforced {keys}"
            )
    assert not mismatches, (
        "advertised metric contract diverges from enforcement (read_metric "
        "would mislead the agent):\n  " + "\n  ".join(mismatches)
    )
