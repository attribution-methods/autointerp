"""Contrastive directions and vector geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .activations import LayerLike, capture_activations
from .model import ModelHandle

DEFAULT_BASELINE_TERMS = [
    "desks", "jackets", "laughter", "bicycles", "chairs", "sand", "pottery",
    "jewelry", "archives", "stars", "traffic", "honey", "ribbons",
    "puzzles", "diamonds", "vinegar", "rivers", "cakes", "glaciers",
    "flowers", "research", "clouds", "bonfires", "time", "gardens",
]


def normalize_vector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / (vector.norm() + eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.flatten()
    b = b.flatten()
    return ((a * b).sum() / (a.norm() * b.norm() + eps)).item()


def contrastive_direction(
    handle: ModelHandle,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer: LayerLike,
    component: str = "resid",
    token_index: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    pos = capture_activations(handle, positive_prompts, layer, component, token_index)
    neg = capture_activations(handle, negative_prompts, layer, component, token_index)
    direction = pos.mean(dim=0) - neg.mean(dim=0)
    return normalize_vector(direction) if normalize else direction


def prompt_from_term(handle: ModelHandle, term: str, template: str = "Tell me about {term}") -> str:
    return handle.format_messages(
        [{"role": "user", "content": template.format(term=term)}],
        add_generation_prompt=True,
    )


def baseline_subtracted_direction(
    handle: ModelHandle,
    concept_term: str,
    layer: LayerLike,
    baseline_terms: Optional[List[str]] = None,
    template: str = "Tell me about {term}",
    token_index: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    baselines = baseline_terms or DEFAULT_BASELINE_TERMS
    concept_prompt = prompt_from_term(handle, concept_term, template)
    baseline_prompts = [prompt_from_term(handle, term, template) for term in baselines]
    concept = capture_activations(handle, [concept_prompt], layer, token_index=token_index)[0]
    baseline = capture_activations(
        handle,
        baseline_prompts,
        layer,
        token_index=token_index,
    ).mean(dim=0)
    direction = concept - baseline
    return normalize_vector(direction) if normalize else direction


def batch_contrastive_directions(
    handle: ModelHandle,
    pairs: Dict[str, Tuple[List[str], List[str]]],
    layer: LayerLike,
    component: str = "resid",
    token_index: int = -1,
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    return {
        name: contrastive_direction(
            handle,
            positives,
            negatives,
            layer,
            component=component,
            token_index=token_index,
            normalize=normalize,
        )
        for name, (positives, negatives) in pairs.items()
    }


def save_vector(vector: torch.Tensor, path: Path, metadata: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector.detach().cpu(), path)
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_vector(path: Path) -> Tuple[torch.Tensor, Optional[dict]]:
    path = Path(path)
    vector = torch.load(path, map_location="cpu")
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return vector, metadata
