"""Model-id resolution: curated aliases + the dynamic Hub-search fallback so a
hand-maintained table never has to keep up with weekly model releases."""

from __future__ import annotations

import pytest

# autointerp.tools.model imports torch at module level — an extra dep CI's core
# install intentionally lacks. Skip there; run wherever torch exists.
pytest.importorskip("torch")

from autointerp.tools import model as M  # noqa: E402


def test_curated_alias_and_passthrough() -> None:
    assert M.resolve_model_id("gpt2-small") == "gpt2"          # irreducible quirk
    assert M.resolve_model_id("Qwen/Qwen3-32B") == "Qwen/Qwen3-32B"  # real id passes through
    assert M.resolve_model_id("some-brand-new-model") == "some-brand-new-model"


def test_placeholder_model_detection() -> None:
    # Real, loadable repo ids are never flagged.
    for ok in ("gpt2", "gpt2-medium", "EleutherAI/pythia-160m",
               "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
               "gpt2-small", "pythia-70m", "gpt2 "):
        assert not M.looks_like_placeholder_model(ok), ok
    # Templates / paths / stand-ins a weak planner emits are all caught.
    for bad in ("local:/path/to/pythia-125M", "<model>", "your-model",
                "/path/to/weights", "model", "TBD", "", "path:foo",
                "file:/x", "model-name", "pythia ...", "a small open model"):
        assert M.looks_like_placeholder_model(bad), bad


def test_missing_repo_detection() -> None:
    assert M._looks_like_missing_repo(Exception("404 Client Error ... tokenizer_config.json"))
    assert M._looks_like_missing_repo(Exception("gpt2-small is not a valid model identifier"))
    assert not M._looks_like_missing_repo(Exception("CUDA out of memory"))


def test_hub_search_picks_exact_basename(monkeypatch) -> None:
    """A bare colloquial name resolves to the canonical repo via an exact
    basename match (highest-downloaded first) — stubbed so it's offline."""
    class _Model:
        def __init__(self, mid: str) -> None:
            self.id = mid

    class _Api:
        def list_models(self, **kw):
            return [
                _Model("foo/pythia-160m-deduped"),   # not an exact basename
                _Model("EleutherAI/pythia-160m"),     # exact -> should win
                _Model("bar/pythia-160m"),
            ]

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    assert M._resolve_via_hub_search("pythia-160m") == "EleutherAI/pythia-160m"


def test_hub_search_none_when_no_exact_match(monkeypatch) -> None:
    class _Model:
        def __init__(self, mid: str) -> None:
            self.id = mid

    class _Api:
        def list_models(self, **kw):
            return [_Model("datificate/gpt2-small-spanish")]  # finetune, not canonical

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    assert M._resolve_via_hub_search("gpt2-small") is None  # falls back to curated alias


def test_gpu_arch_preflight_blocks_unsupported(monkeypatch) -> None:
    cuda = M.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "get_device_capability", lambda i=0: (10, 0))  # sm_100
    monkeypatch.setattr(cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])    # no sm_100
    monkeypatch.setattr(cuda, "get_device_name", lambda i=0: "NVIDIA B200")
    with pytest.raises(RuntimeError, match="SMALLER MODEL WILL NOT HELP"):
        M._assert_gpu_arch_supported("cuda")


def test_gpu_arch_preflight_allows_supported(monkeypatch) -> None:
    cuda = M.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "get_device_capability", lambda i=0: (9, 0))     # sm_90
    monkeypatch.setattr(cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    M._assert_gpu_arch_supported("cuda")  # supported -> no raise


def test_gpu_arch_preflight_cpu_never_raises() -> None:
    M._assert_gpu_arch_supported("cpu")
