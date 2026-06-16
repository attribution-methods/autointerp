"""Model-name aliases and canonicalization — deliberately torch-free.

Split out from :mod:`autointerp.tools.model` (which imports torch + transformers
at module load) so that lightweight layers — spec finalization, validation, the
agent planner — can canonicalize a model name to its Hugging Face repo id
*without* paying a multi-second torch import or hard-depending on a GPU stack.

`autointerp.tools.model` re-exports these names, so existing
``from autointerp.tools.model import resolve_model_id, MODEL_ALIASES`` keeps
working.
"""

from __future__ import annotations

from typing import Dict

MODEL_ALIASES: Dict[str, str] = {
    # Small open models, by the names interp researchers actually use.
    # "gpt2-small" is the TransformerLens / literature name; on the HF Hub the
    # small GPT-2 is just "gpt2" (medium/large/xl exist under their own names,
    # so only "small" needs remapping).
    "gpt2-small": "gpt2",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
}

PREQUANTIZED_ALIASES = {"qwen3-235b"}
MODELS_WITHOUT_SYSTEM_ROLE = {"gemma"}


_PLACEHOLDER_EXACT = {
    "model",
    "model-name",
    "model_name",
    "modelname",
    "your-model",
    "your_model",
    "yourmodel",
    "model-id",
    "model_id",
    "tbd",
    "todo",
    "xxx",
    "none",
}


def looks_like_placeholder_model(model_name: str) -> bool:
    """True if ``model_name`` is a stand-in, not a loadable Hub repo id.

    A weak planner sometimes fills the model field with a template instead of a
    real repo — ``local:/path/to/pythia-125M``, ``<model>``, ``your-model``,
    ``/path/to/weights``. A real Hub id is ``org/name`` or ``name`` over
    ``[A-Za-z0-9._-]`` with at most one slash, so any whitespace, angle bracket,
    URL-ish scheme (``local:``/``file:``/``path:``), ``/path/to/`` segment, ``...``
    ellipsis, or a known stand-in word is a dead giveaway. Catching this at plan
    time turns a mid-run model-load bail into a fixable validation note.
    """
    raw = (model_name or "").strip()
    if not raw:
        return True
    low = raw.lower()
    if low in _PLACEHOLDER_EXACT:
        return True
    if any(ch in raw for ch in "<> \t\n"):
        return True
    if "..." in raw or "/path/to" in low or "placeholder" in low:
        return True
    if low.startswith(("local:", "file:", "path:", "/")):
        return True
    # A scheme-like prefix ("local:foo") that isn't a real "org/name@revision".
    head = low.split(":", 1)[0]
    if ":" in raw and head in {"local", "file", "path", "model", "weights"}:
        return True
    return False


def resolve_model_id(model_name: str) -> str:
    """Map a user/agent model name to a Hugging Face repo id.

    Layered so the static table never has to keep up with the Hub:
      1. a small curated table of colloquial interp names that don't map
         obviously (``gpt2-small`` -> ``gpt2``) or where we prefer a specific
         variant (a short Llama/Qwen name -> its ``-Instruct`` repo);
      2. otherwise pass the name through — a real repo id, including any model
         released this week, loads as-is with no code change;
      3. if that turns out not to be a repo, ``load_model`` searches the Hub
         live (:func:`autointerp.tools.model._resolve_via_hub_search`) for the
         canonical match.
    """
    return MODEL_ALIASES.get(model_name, model_name)
