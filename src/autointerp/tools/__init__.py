"""Shared interpretability tool implementations.

``ModelHandle``/``load_model`` are exposed lazily (PEP 562) so that importing a
*light* submodule — e.g. ``autointerp.tools.model_aliases`` for spec finalize or
validation — does not transitively import torch + transformers via ``.model``.
``from autointerp.tools import load_model`` still works; the torch import just
happens on first access instead of at package import.
"""

from typing import TYPE_CHECKING

__all__ = ["ModelHandle", "load_model"]

if TYPE_CHECKING:  # for type checkers / IDEs only — not executed at runtime
    from .model import ModelHandle, load_model


def __getattr__(name: str):
    if name in {"ModelHandle", "load_model"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
