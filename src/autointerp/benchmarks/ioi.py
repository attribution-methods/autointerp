"""IOI benchmark — Wang-et-al-2022-style indirect object identification.

Pure-Python pair generator + behavioral metric. No model load required at
import time; tokenization is deferred to ``generate_pairs`` and only
needed when the caller passes a tokenizer.

Closed-vocabulary by design: names / places / objects are restricted to a
small list of single-token-with-leading-space tokens for GPT-2's BPE so
that clean and corrupted prompts tokenize to identical length and the
prediction position aligns. The dev / heldout split uses *disjoint* name
pairs to keep validation honest.

Compatible with the :class:`autointerp.benchmarks._protocol.Benchmark`
structural protocol — module-level ``GENERATOR_ID``, ``generate_pairs``,
and ``behavioral_metric``.
"""

from __future__ import annotations

import random
from typing import Any

from ._protocol import StimulusPair


GENERATOR_ID = "ioi_template_generator"


# Closed single-token-with-leading-space vocab for GPT-2 BPE. Each entry
# tokenizes to exactly one BPE token; we verify in ``generate_pairs`` and
# drop any that don't (defensive — these were chosen by the spec to be
# single-token, but the model loader has the final word).
_NAMES_DEV: tuple[str, ...] = (
    " John", " Mary", " Tom", " Sarah", " James", " Kate", " Robert", " Lisa",
    " Michael", " Emma", " David", " Anna", " Peter", " Laura", " Steven", " Rachel",
)
_NAMES_HELDOUT: tuple[str, ...] = (
    " Daniel", " Sophia", " Brian", " Olivia", " Frank", " Hannah", " George", " Emily",
)
_PLACES: tuple[str, ...] = (
    " store", " park", " school", " office", " hospital", " library",
)
_OBJECTS: tuple[str, ...] = (
    " ring", " book", " key", " ball", " pen", " bag",
)
# Additional novel names for the ABC corruption — must be disjoint from
# both _NAMES_DEV and _NAMES_HELDOUT so the corrupted prompt's third name
# is genuinely novel (not just a swap).
_NAMES_C: tuple[str, ...] = (
    " Henry", " Linda", " Mark", " Jenny", " Paul", " Susan",
)


_TEMPLATES_BABA = "When{A} and{B} went to the{place},{B} gave a{object} to"
_TEMPLATES_ABBA = "When{A} and{B} went to the{place},{A} gave a{object} to"


def _verify_single_token(tokenizer, text: str) -> bool:
    """True iff ``tokenizer(text)`` produces exactly one token."""
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception:
        return False
    return isinstance(ids, list) and len(ids) == 1


def _filter_single_token(tokenizer, items: tuple[str, ...]) -> list[str]:
    return [it for it in items if _verify_single_token(tokenizer, it)]


def _tokenize_one(tokenizer, text: str) -> int:
    return int(tokenizer(text, add_special_tokens=False)["input_ids"][0])


def generate_pairs(
    spec: Any,
    *,
    n_pairs: int,
    seed: int,
    tokenizer: Any = None,
    split: str = "dev",
) -> list[StimulusPair]:
    """Build ``n_pairs`` IOI clean/corrupted contrast pairs.

    Parameters
    ----------
    spec:
        The :class:`autointerp.spec.InvestigationSpec` (or any dataset-
        carrying object). Currently used only to read ``dataset.seed`` as
        a tiebreaker when ``seed`` is not explicitly passed; reserved for
        future template overrides.
    n_pairs:
        Target number of pairs. The function returns up to this many; if
        the closed vocab cannot produce that many distinct trials, it
        returns however many it could.
    seed:
        Per-call seed. Determines ABBA/BABA balance, name draws, and
        place/object draws. Should be set per-stage (e.g. dataset.seed
        for dev; a separate seed for heldout to keep splits disjoint).
    tokenizer:
        Optional model tokenizer. When provided, names are filtered to
        ones that tokenize to a single BPE token *with that tokenizer* —
        so the same benchmark adapts cleanly across GPT-2 / Llama / Qwen.
        When ``None``, returns text-only pairs with placeholder ids
        (``-1``); useful for CI smoke tests that don't need the model.
    split:
        ``"dev"`` or ``"heldout"`` — selects the disjoint name pool.
    """
    rng = random.Random(seed)

    name_pool_raw = _NAMES_DEV if split == "dev" else _NAMES_HELDOUT
    if tokenizer is None:
        names = list(name_pool_raw)
        c_names = list(_NAMES_C)
        places = list(_PLACES)
        objects = list(_OBJECTS)
    else:
        names = _filter_single_token(tokenizer, name_pool_raw)
        c_names = _filter_single_token(tokenizer, _NAMES_C)
        places = _filter_single_token(tokenizer, _PLACES)
        objects = _filter_single_token(tokenizer, _OBJECTS)
        if len(names) < 2 or not c_names or not places or not objects:
            raise RuntimeError(
                "IOI benchmark: tokenizer pruned the closed vocab below the "
                "minimum needed (>= 2 names, >= 1 of each other type). "
                f"Have names={names!r}, c={c_names!r}, places={places!r}, "
                f"objects={objects!r}"
            )

    # Balance ABBA / BABA 50:50 across the requested pairs.
    pairs: list[StimulusPair] = []
    for i in range(n_pairs):
        is_abba = (i % 2 == 0)
        # Draw a distinct (A, B) and a third C disjoint from both.
        a_b = rng.sample(names, 2)
        a_name, b_name = a_b
        candidate_cs = [c for c in c_names if c not in (a_name, b_name)]
        if not candidate_cs:
            # Fall back: use a name from the pool that isn't A or B.
            candidate_cs = [n for n in names if n not in (a_name, b_name)]
        c_name = rng.choice(candidate_cs)
        place = rng.choice(places)
        obj = rng.choice(objects)

        tmpl = _TEMPLATES_ABBA if is_abba else _TEMPLATES_BABA
        clean_prompt = tmpl.format(A=a_name, B=b_name, place=place, object=obj)
        # ABC corruption: replace ONE of the repeated-name occurrences with
        # C. By Wang-et-al's convention this is the *non-prediction*
        # occurrence: the first instance for ABBA (A appears at pos 1 in
        # clean; corruption swaps pos 1 to C). For BABA the same logic.
        if is_abba:
            corrupted_prompt = (
                f"When{c_name} and{b_name} went to the{place},{a_name} gave a{obj} to"
            )
        else:
            corrupted_prompt = (
                f"When{c_name} and{b_name} went to the{place},{b_name} gave a{obj} to"
            )

        # IO target = the non-repeated name (always A in our templates).
        # Foil = the repeated name (always B).
        io_name = a_name
        s_name = b_name
        if tokenizer is not None:
            io_id = _tokenize_one(tokenizer, io_name)
            s_id = _tokenize_one(tokenizer, s_name)
        else:
            io_id, s_id = -1, -1

        pairs.append(
            StimulusPair(
                pair_id=f"{split}-{i:04d}",
                clean_prompt=clean_prompt,
                corrupted_prompt=corrupted_prompt,
                target_token_id=io_id,
                foil_token_id=s_id,
                prediction_position=-1,
                metadata={
                    "order": "ABBA" if is_abba else "BABA",
                    "io_name": io_name.strip(),
                    "s_name": s_name.strip(),
                    "c_name": c_name.strip(),
                    "place": place.strip(),
                    "object": obj.strip(),
                    "split": split,
                },
            )
        )
    return pairs


def behavioral_metric(logits: Any, pair: StimulusPair) -> float:
    """``logit(IO) - logit(S)`` at the prediction position. Higher is better.

    ``logits`` may be a 1-D ``[vocab]`` tensor (single trial) or anything
    indexable by ``int`` that returns a Python float at the target /
    foil token positions. Substrate evaluators batch-mean across pairs.
    """
    try:
        # Tensor-like
        target = float(logits[pair.target_token_id])
        foil = float(logits[pair.foil_token_id])
    except Exception as exc:
        raise TypeError(
            f"behavioral_metric: logits must be indexable by int token id; "
            f"got {type(logits).__name__}"
        ) from exc
    return target - foil


__all__ = [
    "GENERATOR_ID",
    "behavioral_metric",
    "generate_pairs",
]
