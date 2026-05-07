"""Tests for the IOI benchmark module."""

from __future__ import annotations

import pytest

from autointerp.benchmarks import (
    BenchmarkNotFoundError,
    StimulusPair,
    get_benchmark,
    list_benchmarks,
)


def test_ioi_is_registered():
    assert "ioi_template_generator" in list_benchmarks()


def test_unknown_benchmark_rejected():
    with pytest.raises(BenchmarkNotFoundError):
        get_benchmark("nope")


def test_ioi_module_loads():
    ioi = get_benchmark("ioi_template_generator")
    assert ioi.GENERATOR_ID == "ioi_template_generator"
    assert callable(ioi.generate_pairs)
    assert callable(ioi.behavioral_metric)


def test_ioi_pairs_are_deterministic_no_tokenizer():
    ioi = get_benchmark("ioi_template_generator")
    a = ioi.generate_pairs(spec=None, n_pairs=6, seed=42, tokenizer=None, split="dev")
    b = ioi.generate_pairs(spec=None, n_pairs=6, seed=42, tokenizer=None, split="dev")
    assert len(a) == len(b) == 6
    for x, y in zip(a, b):
        assert x.clean_prompt == y.clean_prompt
        assert x.corrupted_prompt == y.corrupted_prompt
        assert x.metadata == y.metadata


def test_ioi_pairs_balanced_abba_baba():
    ioi = get_benchmark("ioi_template_generator")
    pairs = ioi.generate_pairs(spec=None, n_pairs=10, seed=0, tokenizer=None, split="dev")
    orders = [p.metadata["order"] for p in pairs]
    assert orders.count("ABBA") == 5
    assert orders.count("BABA") == 5


def test_ioi_dev_and_heldout_disjoint_names():
    ioi = get_benchmark("ioi_template_generator")
    dev = ioi.generate_pairs(spec=None, n_pairs=12, seed=1, tokenizer=None, split="dev")
    held = ioi.generate_pairs(spec=None, n_pairs=12, seed=2, tokenizer=None, split="heldout")
    dev_io = {p.metadata["io_name"] for p in dev}
    dev_s = {p.metadata["s_name"] for p in dev}
    held_io = {p.metadata["io_name"] for p in held}
    held_s = {p.metadata["s_name"] for p in held}
    # Dev and heldout draw from disjoint name pools so contamination
    # isn't possible at the IO/S levels.
    assert dev_io.isdisjoint(held_io)
    assert dev_s.isdisjoint(held_s)


def test_ioi_corruption_keeps_length_implicitly_via_template():
    """Each clean / corrupted pair should have the same number of words
    (a coarse proxy for token-length alignment when no tokenizer is
    available)."""
    ioi = get_benchmark("ioi_template_generator")
    pairs = ioi.generate_pairs(spec=None, n_pairs=8, seed=7, tokenizer=None, split="dev")
    for p in pairs:
        assert len(p.clean_prompt.split()) == len(p.corrupted_prompt.split()), (
            p.clean_prompt, p.corrupted_prompt
        )


def test_behavioral_metric_signs_target_above_foil():
    ioi = get_benchmark("ioi_template_generator")
    pair = StimulusPair(
        pair_id="t",
        clean_prompt="x",
        corrupted_prompt="y",
        target_token_id=3,
        foil_token_id=5,
    )
    # logits where target is higher → positive
    logits = [0.0] * 10
    logits[3] = 1.5
    logits[5] = 0.2
    score_higher = ioi.behavioral_metric(logits, pair)
    assert score_higher > 0
    # flip → negative
    logits[3], logits[5] = 0.2, 1.5
    score_lower = ioi.behavioral_metric(logits, pair)
    assert score_lower < 0
