"""Neuronpedia feature-label lookup.

Two HTTP entry points to https://www.neuronpedia.org:

- ``lookup_neuronpedia_label(release, layer, feature_id)`` — single feature
  by index. Returns the auto-interp description string (or ``None`` if the
  feature is unlabeled).
- ``search_features_by_label(release, layer, query, k)`` — semantic search
  over the labeled feature corpus. Returns ``(feature_id, label, score)``
  tuples ranked by match.

A small file-backed cache lives at ``~/.cache/autointerp/neuronpedia/``
so repeated lookups don't re-hit the API. The cache is content-addressed
on ``(release, layer, feature_id)`` for the per-feature path and on
``(release, layer, query, k)`` for search.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NEURONPEDIA_BASE = "https://www.neuronpedia.org/api"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autointerp" / "neuronpedia"
DEFAULT_TIMEOUT_S = 15


@dataclass(frozen=True)
class FeatureLabel:
    feature_id: int
    label: str
    score: float = 1.0  # 1.0 for direct lookup, <1 for search hits


# ---- HTTP helpers -----------------------------------------------------------


def _cache_dir() -> Path:
    override = os.environ.get("AUTOINTERP_NEURONPEDIA_CACHE")
    return Path(override) if override else DEFAULT_CACHE_DIR


def _cache_path(*key_parts: str) -> Path:
    digest = hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()[:16]
    return _cache_dir() / f"{digest}.json"


def _read_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(value))
    except OSError:
        pass


def _http_get(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> Any:
    import requests  # lazy — keeps the dep optional at top-level

    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    headers = {"X-Api-Key": api_key} if api_key else {}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---- Public API -------------------------------------------------------------


def lookup_neuronpedia_label(
    release: str,
    layer: int,
    feature_id: int,
    *,
    use_cache: bool = True,
) -> str | None:
    """Return the auto-interp description for ``(release, layer, feature_id)``.

    ``release`` is the Neuronpedia model identifier (e.g. ``gemma-2-9b``).
    Returns ``None`` if the feature has no label or the API call fails —
    callers should treat ``None`` as "unknown", not as a hard error.
    """
    key = (release, str(layer), str(feature_id))
    cache_path = _cache_path("label", *key) if use_cache else None
    if cache_path is not None:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached.get("label")

    url = f"{NEURONPEDIA_BASE}/feature/{release}/{layer}-res/{feature_id}"
    try:
        data = _http_get(url)
    except Exception:
        return None
    label = _extract_label(data)
    if cache_path is not None:
        _write_cache(cache_path, {"label": label})
    return label


def search_features_by_label(
    release: str,
    layer: int,
    query: str,
    *,
    k: int = 20,
    use_cache: bool = True,
) -> list[FeatureLabel]:
    """Semantic search over labeled features on Neuronpedia.

    Returns up to ``k`` features ranked by the API's relevance score.
    Empty list on failure or no hits.
    """
    key = (release, str(layer), query, str(k))
    cache_path = _cache_path("search", *key) if use_cache else None
    if cache_path is not None:
        cached = _read_cache(cache_path)
        if cached is not None:
            return [FeatureLabel(**row) for row in cached]

    url = (
        f"{NEURONPEDIA_BASE}/explanation/search"
        f"?modelId={release}&layers={layer}-res&query={query}&maxResults={k}"
    )
    try:
        data = _http_get(url)
    except Exception:
        return []
    rows = _extract_search_rows(data)
    if cache_path is not None:
        _write_cache(cache_path, [row.__dict__ for row in rows])
    return rows


# ---- Response parsers (Neuronpedia's payload shape) -------------------------


def _extract_label(data: Any) -> str | None:
    """Pluck the auto-interp description from a feature payload.

    Neuronpedia's response shape varies by endpoint version; tolerate
    both ``explanations[0].description`` and a flat ``description``.
    """
    if not isinstance(data, dict):
        return None
    explanations = data.get("explanations")
    if isinstance(explanations, list) and explanations:
        first = explanations[0]
        if isinstance(first, dict):
            desc = first.get("description")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
    desc = data.get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    return None


def _extract_search_rows(data: Any) -> list[FeatureLabel]:
    if not isinstance(data, list):
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        data = results
    out: list[FeatureLabel] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            fid = int(row.get("index", row.get("feature", -1)))
        except (TypeError, ValueError):
            continue
        if fid < 0:
            continue
        label = (
            row.get("description")
            or row.get("label")
            or (row.get("explanations") or [{}])[0].get("description", "")
        )
        score = float(row.get("score", row.get("similarity", 1.0)) or 1.0)
        if not label:
            continue
        out.append(FeatureLabel(feature_id=fid, label=str(label).strip(), score=score))
    return out


__all__ = [
    "FeatureLabel",
    "NEURONPEDIA_BASE",
    "lookup_neuronpedia_label",
    "search_features_by_label",
]
