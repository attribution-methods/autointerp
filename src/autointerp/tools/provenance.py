"""Capture ledger — input-provenance v2.

A *capture* records that some data (logits, activations, probe scores, …) was
produced by an actual model run, so a metric computed on it is traceable to a
real forward pass rather than hand-typed numbers. Each capture writes its data
under ``<run>/captures/<id>.json`` and appends a tamper-evident line to
``<run>/captures.jsonl``:

    {capture_id, name, sha256, n, source, model_id, prompt_batch, code_ref,
     data_relpath, created_at?}

``source`` is the lineage claim:
  - ``model_forward`` — the value is the output of a real forward pass;
  - ``derived``       — computed from other captures;
  - ``manual``        — typed by the agent (NOT a real measurement).

The agent calls :func:`record_capture` inside its scripts (the run is resolved
from the ``AUTOINTERP_RUN_DIR`` env var the investigation runtime sets). It
returns the relpath to pass as a metric's ``inputs``. The pipeline uses the
run-dir helpers (:func:`capture_for_relpath`, :func:`verify_capture`) to check
that a metric's inputs are a genuine, unmodified capture.

This is an audit-and-traceability layer, not a cryptographic guarantee: the
agent writes the code, so a *deceptive* agent could tag fabricated data
``model_forward``. The value is that it makes the lazy fabrication path
impossible, forces any cheat to leave an inconsistent paper trail
(model_id / n / code_ref), and gives the grounding check a precise signal.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

CAPTURE_SOURCES = ("model_forward", "derived", "manual")
RUN_DIR_ENV = "AUTOINTERP_RUN_DIR"
_CAPTURES_SUBDIR = "captures"
_LEDGER_NAME = "captures.jsonl"


def _ledger_path(run_dir: Path) -> Path:
    return run_dir / _LEDGER_NAME


def _captures_dir(run_dir: Path) -> Path:
    return run_dir / _CAPTURES_SUBDIR


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _sha256(blob: bytes) -> str:
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _infer_n(value: Any) -> int | None:
    """Best-effort sample count, so a criterion's ``n_min`` can be checked
    against the data the metric actually saw."""
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        lens = [len(v) for v in value.values() if isinstance(v, list)]
        return max(lens) if lens else None
    return None


def _has_measurement(value: Any) -> bool:
    """True iff ``value`` carries actual measurement data — the kind a forward
    pass produces: a number, a non-empty array, or generated text. False for
    empty containers (``[]``, ``{}``, ``{"prompts": []}``) and label-only stubs
    (``{"placeholder": True, "notes": "..."}``). Used to reject the placeholder
    capture an agent writes to satisfy the gate without running the model.

    A ``bool`` is NOT a measurement (it is a flag); a ``str`` alone is not (a
    note), but a non-empty list of strings IS (generated text). Numbers and
    nested structures that themselves contain measurements count.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str) or value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, dict):
        return any(_has_measurement(v) for v in value.values())
    return False


def record_to(
    run_dir: str | Path,
    name: str,
    value: Any,
    *,
    source: str = "model_forward",
    model_id: str | None = None,
    n: int | None = None,
    prompt_batch: str | None = None,
    code_ref: str | None = None,
) -> str:
    """Write a capture under ``run_dir`` and append its ledger entry. Returns
    the relpath (e.g. ``captures/cap_ab12cd34.json``) to pass as metric inputs."""
    run_dir = Path(run_dir)
    if source not in CAPTURE_SOURCES:
        raise ValueError(f"source must be one of {CAPTURE_SOURCES}, got {source!r}")
    if not isinstance(name, str) or not name:
        raise ValueError("capture name (non-empty string) is required")
    if source == "model_forward" and not _has_measurement(value):
        raise ValueError(
            f"model_forward capture {name!r} carries no measurement data "
            "(empty or placeholder). A real forward pass produces at least one "
            "row of numbers/logits/predictions or generated text. Run the model "
            "on real prompts and capture the actual outputs — do not write a "
            "placeholder/stub capture to satisfy the gate. (If the value is "
            "genuinely hand-specified and not a measurement, use source='manual', "
            "but then it cannot ground a causal/behavioral criterion.)"
        )
    cdir = _captures_dir(run_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    blob = _canonical_bytes(value)
    cap_id = "cap_" + secrets.token_hex(8)
    data_relpath = f"{_CAPTURES_SUBDIR}/{cap_id}.json"
    (run_dir / data_relpath).write_bytes(blob)
    record = {
        "capture_id": cap_id,
        "name": name,
        "sha256": _sha256(blob),
        "n": _infer_n(value) if n is None else n,
        "source": source,
        "model_id": model_id,
        "prompt_batch": prompt_batch,
        "code_ref": code_ref,
        "data_relpath": data_relpath,
    }
    with _ledger_path(run_dir).open("a") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    return data_relpath


def record_capture(name: str, value: Any, **kwargs: Any) -> str:
    """Agent-facing: record a capture in the current investigation run. The run
    is resolved from ``AUTOINTERP_RUN_DIR`` (set by the runtime when your bash
    script executes). Returns the relpath to pass as a metric's ``inputs``."""
    run_dir = os.environ.get(RUN_DIR_ENV)
    if not run_dir:
        raise RuntimeError(
            f"record_capture must run inside an investigation: {RUN_DIR_ENV} is "
            "not set. (It is set automatically when the agent runs a script.)"
        )
    return record_to(run_dir, name, value, **kwargs)


def load_ledger(run_dir: str | Path) -> dict[str, dict]:
    """Return ``{capture_id: record}`` for every capture recorded in this run."""
    path = _ledger_path(Path(run_dir))
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = rec.get("capture_id")
        if cid:
            out[cid] = rec
    return out


def capture_for_relpath(run_dir: str | Path, relpath: str) -> dict | None:
    """Find the ledger record whose data file is ``relpath`` (as the agent would
    pass it to a metric). Returns None if the path is not a recorded capture."""
    norm = str(relpath).replace("\\", "/").lstrip("./")
    for rec in load_ledger(run_dir).values():
        if rec.get("data_relpath") == norm:
            return rec
    return None


def verify_capture(run_dir: str | Path, record: dict) -> bool:
    """Re-hash the capture's data file and confirm it matches the ledger — i.e.
    the data was not swapped out after it was recorded."""
    rel = record.get("data_relpath")
    if not rel:
        return False
    data_path = Path(run_dir) / rel
    if not data_path.exists():
        return False
    try:
        return _sha256(data_path.read_bytes()) == record.get("sha256")
    except OSError:
        return False


__all__ = [
    "CAPTURE_SOURCES",
    "RUN_DIR_ENV",
    "capture_for_relpath",
    "load_ledger",
    "record_capture",
    "record_to",
    "verify_capture",
]
