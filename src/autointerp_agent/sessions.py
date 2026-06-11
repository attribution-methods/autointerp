"""Per-project session persistence for the interactive shell.

Claude-Code-style semantics: ``autointerp`` always starts a fresh session
(any stray spec draft is archived, never silently resumed); ``autointerp
--continue`` opens a picker over this project's previous sessions and
restores the chosen one — conversation messages, spec draft, turn counter,
and cost — so the agent continues with full context.

Sessions live at ``~/.autointerp/sessions/<cwd-slug>/<session-id>.json``:
keyed by working directory (like Claude Code's per-project transcripts) but
stored under the user dir so project trees stay clean.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autointerp.utils.age import iso_age_string
from autointerp.utils.cost import CostTracker, ModelUsage

from .config import USER_DIR

SESSIONS_ROOT = USER_DIR / "sessions"
DRAFT_NAME = "_draft.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cwd_slug(cwd: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(cwd)).strip("-") or "root"


@dataclass
class SessionMeta:
    session_id: str
    title: str
    turn: int
    model_name: str
    updated_at: str
    path: Path

    def age(self) -> str:
        return iso_age_string(self.updated_at)


class SessionStore:
    """Atomic JSON persistence for one project's interactive sessions."""

    def __init__(self, cwd: Path | None = None, root: Path | None = None) -> None:
        base = root or SESSIONS_ROOT
        self.dir = base / _cwd_slug(cwd or Path.cwd())

    @staticmethod
    def new_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:4]}"

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(
        self,
        session_id: str,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        turn: int,
        cost: dict[str, Any],
        title: str,
        draft: str | None,
    ) -> Path:
        path = self._path(session_id)
        created_at = _now_iso()
        if path.exists():
            try:
                created_at = json.loads(path.read_text()).get("created_at", created_at)
            except (json.JSONDecodeError, OSError):
                pass
        payload = {
            "session_id": session_id,
            "title": title,
            "created_at": created_at,
            "updated_at": _now_iso(),
            "cwd": str(Path.cwd()),
            "model_name": model_name,
            "turn": turn,
            "messages": messages,
            "cost": cost,
            "draft": draft,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        tmp.replace(path)
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def list(self) -> list[SessionMeta]:
        """All sessions in this project, newest first. Corrupt files skipped."""
        if not self.dir.exists():
            return []
        metas: list[SessionMeta] = []
        for path in self.dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not data.get("session_id"):
                continue
            metas.append(
                SessionMeta(
                    session_id=str(data["session_id"]),
                    title=str(data.get("title") or "(untitled)"),
                    turn=int(data.get("turn") or 0),
                    model_name=str(data.get("model_name") or "?"),
                    updated_at=str(data.get("updated_at") or ""),
                    path=path,
                )
            )
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas


# ---------------------------------------------------------------------------
# Spec-draft pairing (the draft belongs to its session)
# ---------------------------------------------------------------------------


def snapshot_draft(spec_dir: Path) -> str | None:
    draft = spec_dir / DRAFT_NAME
    try:
        return draft.read_text() if draft.exists() else None
    except OSError:
        return None


def restore_draft(spec_dir: Path, draft_text: str | None) -> None:
    if draft_text is None:
        return
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / DRAFT_NAME).write_text(draft_text)


def archive_stray_draft(spec_dir: Path) -> Path | None:
    """Fresh sessions never inherit an old draft — move it aside, keep data."""
    draft = spec_dir / DRAFT_NAME
    if not draft.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = spec_dir / f"_draft-{stamp}.bak.json"
    try:
        draft.rename(target)
    except OSError:
        return None
    return target


# ---------------------------------------------------------------------------
# Cost restoration
# ---------------------------------------------------------------------------


def cost_from_dict(data: dict[str, Any] | None, run_id: str) -> CostTracker:
    tracker = CostTracker(run_id=run_id)
    if not data:
        return tracker
    tracker.total_cost_usd = float(data.get("total_cost_usd", 0.0))
    tracker.total_requests = int(data.get("total_requests", 0))
    tracker.has_unknown_cost = bool(data.get("has_unknown_cost", False))
    for name, raw in (data.get("by_model") or {}).items():
        tracker.by_model[name] = ModelUsage(
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
            cache_read_input_tokens=int(raw.get("cache_read_input_tokens", 0)),
            cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens", 0)),
            requests=int(raw.get("requests", 0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
        )
    return tracker


# ---------------------------------------------------------------------------
# The --continue picker
# ---------------------------------------------------------------------------


async def pick_session(store: SessionStore, console: Any) -> dict[str, Any] | None:
    """Arrow-key picker over this project's sessions. None = start fresh."""
    from .model_select import _select_async

    metas = store.list()
    if not metas:
        console.print(
            "[yellow]No previous sessions in this project — starting fresh.[/yellow]"
        )
        return None
    options = [
        (
            meta.title[:64],
            f"— {meta.turn} turn{'s' if meta.turn != 1 else ''} · "
            f"{meta.model_name} · {meta.age()}",
        )
        for meta in metas
    ]
    choice = await _select_async("Resume a session", options, default_index=0)
    if choice is None:
        console.print("[dim]starting fresh[/dim]")
        return None
    return store.load(metas[choice].session_id)


__all__ = [
    "DRAFT_NAME",
    "SessionMeta",
    "SessionStore",
    "archive_stray_draft",
    "cost_from_dict",
    "pick_session",
    "restore_draft",
    "snapshot_draft",
]
