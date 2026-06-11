"""Tests for per-project session persistence (autointerp_agent.sessions)."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from rich.console import Console

from autointerp_agent import sessions


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def _store(tmp_path: Path, cwd: str = "/proj/a") -> sessions.SessionStore:
    return sessions.SessionStore(cwd=Path(cwd), root=tmp_path / "root")


def _save(store: sessions.SessionStore, session_id: str, *, title: str = "t",
          turn: int = 1) -> None:
    store.save(
        session_id,
        model_name="m",
        messages=[{"role": "user", "content": title}],
        turn=turn,
        cost={"total_cost_usd": 0.5, "total_requests": 2, "by_model": {}},
        title=title,
        draft='{"question": "q"}',
    )


def test_store_round_trip_preserves_created_at(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(store, "s1", title="how does IOI work?")
    first = store.load("s1")
    assert first is not None
    assert first["title"] == "how does IOI work?"
    assert first["draft"] == '{"question": "q"}'
    created = first["created_at"]
    _save(store, "s1", title="how does IOI work?", turn=5)  # update
    second = store.load("s1")
    assert second["turn"] == 5
    assert second["created_at"] == created  # creation time survives updates


def test_list_newest_first_and_tolerates_corruption(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(store, "20260101-000000-aaaa")
    _save(store, "20260102-000000-bbbb")
    # Hand-bump the second file's updated_at so ordering is deterministic.
    p = store.dir / "20260102-000000-bbbb.json"
    data = json.loads(p.read_text())
    data["updated_at"] = "2099-01-01T00:00:00Z"
    p.write_text(json.dumps(data))
    (store.dir / "junk.json").write_text("{not json")
    metas = store.list()
    assert [m.session_id for m in metas] == [
        "20260102-000000-bbbb", "20260101-000000-aaaa"
    ]


def test_stores_are_isolated_per_project(tmp_path: Path) -> None:
    a = _store(tmp_path, "/proj/a")
    b = _store(tmp_path, "/proj/b")
    _save(a, "s1")
    assert a.list() and not b.list()


def test_archive_stray_draft(tmp_path: Path) -> None:
    assert sessions.archive_stray_draft(tmp_path) is None
    draft = tmp_path / sessions.DRAFT_NAME
    draft.write_text('{"question": "old"}')
    archived = sessions.archive_stray_draft(tmp_path)
    assert archived is not None and archived.exists()
    assert archived.name.startswith("_draft-") and archived.name.endswith(".bak.json")
    assert not draft.exists()


def test_snapshot_and_restore_draft(tmp_path: Path) -> None:
    assert sessions.snapshot_draft(tmp_path) is None
    sessions.restore_draft(tmp_path, '{"question": "q"}')
    assert sessions.snapshot_draft(tmp_path) == '{"question": "q"}'
    sessions.restore_draft(tmp_path, None)  # no-op
    assert sessions.snapshot_draft(tmp_path) == '{"question": "q"}'


def test_cost_from_dict_restores_totals() -> None:
    tracker = sessions.cost_from_dict(
        {
            "total_cost_usd": 1.25,
            "total_requests": 7,
            "has_unknown_cost": True,
            "by_model": {"m": {"input_tokens": 100, "output_tokens": 20,
                               "requests": 7, "cost_usd": 1.25}},
        },
        run_id="r",
    )
    assert tracker.total_cost_usd == 1.25 and tracker.total_requests == 7
    assert tracker.has_unknown_cost
    assert tracker.by_model["m"].input_tokens == 100
    empty = sessions.cost_from_dict(None, run_id="r")
    assert empty.total_requests == 0


def test_pick_session_empty_store(tmp_path: Path) -> None:
    console = _console()
    out = asyncio.run(sessions.pick_session(_store(tmp_path), console))
    assert out is None
    assert "No previous sessions" in console.file.getvalue()


def test_pick_session_returns_chosen_payload(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    _save(store, "s1", title="surprise circuits")

    async def fake_select(title, options, default_index=0):
        assert "Resume" in title
        assert "surprise circuits" in options[0][0]
        return 0

    import autointerp_agent.model_select as ms

    monkeypatch.setattr(ms, "_select_async", fake_select)
    payload = asyncio.run(sessions.pick_session(store, _console()))
    assert payload is not None and payload["session_id"] == "s1"


def test_pick_session_escape_starts_fresh(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    _save(store, "s1")

    async def fake_select(title, options, default_index=0):
        return None

    import autointerp_agent.model_select as ms

    monkeypatch.setattr(ms, "_select_async", fake_select)
    console = _console()
    assert asyncio.run(sessions.pick_session(store, console)) is None
    assert "starting fresh" in console.file.getvalue()
