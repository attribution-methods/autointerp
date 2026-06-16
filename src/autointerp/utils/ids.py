"""Short, case-insensitive-safe ID generator.

From claude-code's ``Task.ts`` ID alphabet: digits + lowercase only, so
the resulting IDs are safe on case-insensitive filesystems (macOS HFS+,
many Windows configs) and resist symlink attacks that depend on case
collisions. 36⁸ = ~2.8 trillion combinations, sufficient for run / cache /
finding IDs.
"""

from __future__ import annotations

import secrets

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def generate_id(prefix: str = "", *, length: int = 8) -> str:
    """Return ``prefix + N random base36 chars`` (no separator).

    Use a one-letter prefix to keep the type tag inline (e.g. ``r`` for run,
    ``c`` for cache, ``f`` for finding). Avoid uppercase prefixes — they
    defeat the case-insensitive-safe property.
    """
    if any(c.isupper() for c in prefix):
        raise ValueError(f"id prefix must be lowercase, got {prefix!r}")
    body = "".join(secrets.choice(_ALPHABET) for _ in range(length))
    return prefix + body


__all__ = ["generate_id"]
