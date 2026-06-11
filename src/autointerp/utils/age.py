"""Human-readable age strings for stale data caveats.

From claude-code's ``memdir/memoryAge.ts``: a raw ISO timestamp doesn't
trigger staleness reasoning the way ``"47 days ago"`` does. Use these
on cross-run findings and resume notices so the agent treats old context
as suspect rather than authoritative.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

_DAY_SECONDS = 86_400


def age_days(mtime_seconds: float, *, now: float | None = None) -> int:
    """Floor-rounded days since ``mtime_seconds``. Negative inputs clamp to 0."""
    now_t = time.time() if now is None else now
    return max(0, int((now_t - mtime_seconds) // _DAY_SECONDS))


def age_string(mtime_seconds: float, *, now: float | None = None) -> str:
    """``today`` / ``yesterday`` / ``N days ago``."""
    d = age_days(mtime_seconds, now=now)
    if d == 0:
        return "today"
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


def freshness_caveat(mtime_seconds: float, *, now: float | None = None) -> str:
    """Caveat string for >1d old items. Empty for fresh items.

    Wrap the consumer's text in this when reading from across-run findings,
    cached run summaries, or any artifact whose ground truth could have
    drifted under the agent's feet.
    """
    d = age_days(mtime_seconds, now=now)
    if d <= 1:
        return ""
    return (
        f"This artifact is {d} days old. "
        f"Memories are point-in-time observations, not live state — "
        f"claims about code, models, or measurements may be outdated. "
        f"Verify against current state before treating as fact."
    )


def iso_age_string(iso: str, *, now: float | None = None) -> str:
    """Convenience wrapper for ISO-8601 timestamps (the format we persist)."""
    if not iso:
        return "unknown"
    try:
        # Tolerate both the trailing-Z form we write and full offsets.
        s = iso.replace("Z", "+00:00")
        ts = datetime.fromisoformat(s).astimezone(timezone.utc).timestamp()
    except ValueError:
        return "unknown"
    return age_string(ts, now=now)


__all__ = ["age_days", "age_string", "freshness_caveat", "iso_age_string"]
