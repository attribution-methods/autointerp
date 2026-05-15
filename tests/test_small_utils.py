"""Tests for small lifted helpers: ids, age strings, terminal-state predicate."""

from __future__ import annotations

import re
import time

import pytest

from autointerp.pipelines.investigation.state import (
    TerminalState,
    is_terminal_state,
)
from autointerp.utils.age import age_days, age_string, freshness_caveat, iso_age_string
from autointerp.utils.ids import generate_id


def test_generate_id_shape() -> None:
    rid = generate_id("r", length=8)
    assert rid.startswith("r")
    assert len(rid) == 9
    # Body uses only digits + lowercase — case-insensitive-safe.
    assert re.fullmatch(r"r[0-9a-z]{8}", rid)


def test_generate_id_unique() -> None:
    ids = {generate_id("c") for _ in range(1000)}
    # Vanishingly unlikely to collide at 36^8 space.
    assert len(ids) == 1000


def test_generate_id_rejects_uppercase_prefix() -> None:
    with pytest.raises(ValueError):
        generate_id("Run")


def test_age_days_basic() -> None:
    now = 1_000_000.0
    assert age_days(now, now=now) == 0
    assert age_days(now - 86_400, now=now) == 1
    assert age_days(now - 86_400 * 5, now=now) == 5
    # Future timestamps clamp to 0 (clock skew).
    assert age_days(now + 1000, now=now) == 0


def test_age_string() -> None:
    now = time.time()
    assert age_string(now, now=now) == "today"
    assert age_string(now - 86_400, now=now) == "yesterday"
    assert age_string(now - 47 * 86_400, now=now) == "47 days ago"


def test_freshness_caveat_only_for_old() -> None:
    now = time.time()
    assert freshness_caveat(now, now=now) == ""
    assert freshness_caveat(now - 86_400, now=now) == ""
    caveat = freshness_caveat(now - 14 * 86_400, now=now)
    assert "14 days old" in caveat


def test_iso_age_string_round_trips_our_format() -> None:
    # Matches the format now_iso() emits: "...Z".
    now = time.time()
    iso = "2026-05-04T13:53:41.763583Z"
    out = iso_age_string(iso, now=now)
    assert out in ("today", "yesterday") or "days ago" in out
    assert iso_age_string("not-an-iso") == "unknown"


def test_is_terminal_state_predicate() -> None:
    assert is_terminal_state(None) is False
    assert is_terminal_state(TerminalState.COMPLETED) is True
    assert is_terminal_state(TerminalState.CRITERION_FAILED) is True
    assert is_terminal_state(TerminalState.ABORTED) is True
    assert is_terminal_state(TerminalState.BUDGET_EXHAUSTED) is True
    assert is_terminal_state(TerminalState.REVISION_REQUESTED) is True
