"""Path normalization + scope validation.

Two helpers used by the investigation pipeline's range-restricted ``write_file``:

- ``normalize_safe_path`` rejects relative paths, near-root absolutes (``/``),
  null bytes, and UNC/Windows drive roots; returns an NFC-normalized absolute
  ``Path``. Mirrors the rejection set in Anthropic's ``validateMemoryPath``.
- ``is_within`` decides whether a target path is under any of a set of
  allowed roots, after resolving symlinks. Use for the per-tool write gate.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path


def normalize_safe_path(raw: str | os.PathLike[str]) -> Path | None:
    """Return an NFC-normalized absolute path, or ``None`` if the input is
    structurally unsafe.

    Rejected:
    - empty / None-ish
    - relative (must be absolute after the caller's expansion)
    - too short to be meaningful (``/``, ``/a``)
    - contains a NUL byte (would truncate in syscalls but survive ``Path``)
    - UNC paths (``\\\\server\\share``) — opaque trust boundary
    - Windows drive root (``C:\\``) — write to a whole drive
    """
    if raw is None:
        return None
    s = str(raw)
    if not s or "\0" in s:
        return None
    # Reject UNC / Windows drive roots even on Linux — the inputs may have
    # come from a config file authored on Windows.
    if s.startswith("\\\\") or s.startswith("//"):
        return None
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha() and s.rstrip("\\/") == s[:2]:
        return None
    p = Path(s)
    if not p.is_absolute():
        return None
    # ``Path.resolve(strict=False)`` collapses .. segments without requiring
    # the path to exist (writes target paths that don't exist yet).
    resolved = p.resolve()
    if len(str(resolved)) < 3:
        return None
    # NFC-normalize the string form so equivalent unicode paths hash equal.
    return Path(unicodedata.normalize("NFC", str(resolved)))


def is_within(path: str | os.PathLike[str], roots: list[Path]) -> bool:
    """True iff ``path`` is the same as or a descendant of any root.

    Both ``path`` and each root are resolved (symlinks followed) so that a
    symlink under ``scratch/`` pointing at ``/etc`` is correctly rejected.
    A root that is a *file* matches only its exact path.
    """
    target = Path(path).resolve()
    for root in roots:
        rroot = Path(root).resolve()
        if rroot.is_file() or not rroot.exists():
            # File targets (e.g. INVESTIGATION_LOG.md) must match exactly.
            # Roots that don't exist yet are treated as directory roots; the
            # scaffolding will create them.
            if rroot == target:
                return True
            try:
                target.relative_to(rroot)
                return True
            except ValueError:
                continue
        else:
            try:
                target.relative_to(rroot)
                return True
            except ValueError:
                continue
    return False


__all__ = ["normalize_safe_path", "is_within"]
