"""IOI Stage 0 prompt generation and behavioral scoring helpers."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

import torch

from autointerp.tools.model import load_model


NAMES = [
    " John",
    " Mary",
    " Tom",
    " Sarah",
    " James",
    " Kate",
    " Robert",
    " Lisa",
    " Mark",
    " Anna",
    " Paul",
    " Laura",
    " Mike",
    " Emma",
    " David",
    " Julia",
    " Chris",
    " Amy",
    " Brian",
    " Rachel",
    " Kevin",
    " Chloe",
    " Megan",
    " Peter",
    " Alice",
    " Linda",
    " Susan",
    " Daniel",
    " Nancy",
]
PLACES = [" park", " store", " library", " school", " office", " cafe", " mall", " museum"]
OBJECTS = [" book", " ball", " gift", " note", " package", " letter", " flower", " apple"]


def single_token_values(tokenizer: Any, values: list[str]) -> list[str]:
    """Keep values that are exactly one GPT-style token."""
    return [v for v in values if len(tokenizer.encode(v, add_special_tokens=False)) == 1]


def generate_ioi_dev_prompts(
    tokenizer: Any,
    *,
    seed: int = 42,
    pair_start: int = 50,
    pair_stop: int = 300,
) -> tuple[list[str], list[str], list[str]]:
    """Generate the 500-sample IOI dev split with corrected ABBA/BABA labels."""
    records = generate_ioi_dev_records(
        tokenizer,
        seed=seed,
        pair_start=pair_start,
        pair_stop=pair_stop,
    )
    prompts = [record["prompt"] for record in records]
    io_tokens = [record["io"] for record in records]
    s_tokens = [record["s"] for record in records]
    return prompts, io_tokens, s_tokens


def generate_ioi_dev_records(
    tokenizer: Any,
    *,
    seed: int = 42,
    pair_start: int = 50,
    pair_stop: int = 300,
) -> list[dict[str, Any]]:
    """Generate IOI clean/corrupt records for Stage 0/1 artifacts."""
    names = single_token_values(tokenizer, NAMES)
    places = single_token_values(tokenizer, PLACES)
    objects = single_token_values(tokenizer, OBJECTS)

    rng = random.Random(seed)
    pairs = list(itertools.combinations(names, 2))
    rng.shuffle(pairs)
    dev_pairs = pairs[pair_start:pair_stop]

    records: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(dev_pairs):
        place = rng.choice(places)
        obj = rng.choice(objects)
        c_candidates = [name for name in names if name not in {a, b}]
        c = c_candidates[(idx * 7) % len(c_candidates)]

        # ABBA: repeated subject is B, indirect object is A.
        records.append(
            {
                "template": "ABBA",
                "prompt": f"When{a} and{b} went to the{place},{b} gave a{obj} to",
                "corrupted_prompt": f"When{a} and{c} went to the{place},{b} gave a{obj} to",
                "io": a,
                "s": b,
                "A": a,
                "B": b,
                "C": c,
                "place": place,
                "object": obj,
            }
        )
        # BABA: repeated subject is A, indirect object is B.
        records.append(
            {
                "template": "BABA",
                "prompt": f"When{a} and{b} went to the{place},{a} gave a{obj} to",
                "corrupted_prompt": f"When{c} and{b} went to the{place},{a} gave a{obj} to",
                "io": b,
                "s": a,
                "A": a,
                "B": b,
                "C": c,
                "place": place,
                "object": obj,
            }
        )

    return records


def count_single_token_names(tokenizer: Any) -> int:
    """Return the usable single-token IOI name count for reporting."""
    return len(single_token_values(tokenizer, NAMES))


def run_ioi_stage0_forward(
    *,
    model_name: str = "gpt2",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 16,
    local_files_only: bool = True,
    include_raw: bool = False,
    include_prompt_records: bool = False,
) -> dict[str, Any]:
    """Run GPT-2-small on the IOI dev split and return compact metric inputs."""
    handle = load_model(
        model_name,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    tok = handle.tokenizer
    records = generate_ioi_dev_records(tok)
    prompts = [record["prompt"] for record in records]
    io_tokens = [record["io"] for record in records]
    s_tokens = [record["s"] for record in records]

    io_ids = torch.tensor(
        [tok.encode(x, add_special_tokens=False)[0] for x in io_tokens],
        device=handle.input_device(),
    )
    s_ids = torch.tensor(
        [tok.encode(x, add_special_tokens=False)[0] for x in s_tokens],
        device=handle.input_device(),
    )

    diffs: list[torch.Tensor] = []
    target_logits: list[float] = []
    foil_logits: list[float] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        inputs = tok(chunk, return_tensors="pt", padding=True).to(handle.input_device())
        with torch.inference_mode():
            logits = handle.model(**inputs).logits[:, -1, :].float()
        rows = torch.arange(len(chunk), device=logits.device)
        end = start + len(chunk)
        target = logits[rows, io_ids[start:end]]
        foil = logits[rows, s_ids[start:end]]
        diff = target - foil
        if include_raw:
            target_logits.extend(target.detach().cpu().tolist())
            foil_logits.extend(foil.detach().cpu().tolist())
        diffs.append(diff.detach().cpu())

    all_diffs = torch.cat(diffs)
    n_correct = int((all_diffs > 0).sum().item())
    n_total = int(all_diffs.numel())
    accuracy = n_correct / n_total
    mean_diff = float(all_diffs.mean().item())

    out: dict[str, Any] = {
        "summary": {
            "accuracy": accuracy,
            "mean_logit_diff": mean_diff,
            "n": n_total,
            "n_correct": n_correct,
            "n_names": count_single_token_names(tok),
            "first_prompt": prompts[0] if prompts else None,
            "first_diff": float(all_diffs[0].item()) if n_total else None,
        },
        "compact_accuracy_inputs": {
            "n_correct": n_correct,
            "n_total": n_total,
        },
        "compact_logit_diff_inputs": {
            "mean_diff": mean_diff,
            "n_total": n_total,
        },
    }
    if include_raw:
        out["accuracy_inputs"] = {
            "predictions": [bool(x > 0) for x in all_diffs.tolist()],
            "labels": [True] * n_total,
        }
        out["logit_diff_inputs"] = {
            "target_logits": target_logits,
            "foil_logits": foil_logits,
        }
    if include_prompt_records:
        out["prompt_records"] = records
    return out


def write_metric_inputs(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write Stage 0 metric inputs as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    return out_path


__all__ = [
    "generate_ioi_dev_prompts",
    "generate_ioi_dev_records",
    "count_single_token_names",
    "run_ioi_stage0_forward",
    "single_token_values",
    "write_metric_inputs",
]
