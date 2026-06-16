"""Model loading, chat formatting, and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Aliases + canonicalization live in a torch-free module so lightweight layers
# (spec finalize, validation, planner) can canonicalize without importing torch.
# Re-exported here for back-compat: existing `from autointerp.tools.model import
# resolve_model_id, MODEL_ALIASES` keeps working.
from .model_aliases import MODEL_ALIASES as MODEL_ALIASES  # re-export  # noqa: E402
from .model_aliases import (  # noqa: E402
    MODELS_WITHOUT_SYSTEM_ROLE,
    PREQUANTIZED_ALIASES,
    resolve_model_id,
)
from .model_aliases import (  # noqa: E402  # re-export for external callers
    looks_like_placeholder_model as looks_like_placeholder_model,
)


def _resolve_via_hub_search(model_name: str) -> Optional[str]:
    """Dynamic fallback: find the canonical Hub repo for a bare/colloquial name
    (e.g. ``pythia-160m`` -> ``EleutherAI/pythia-160m``). Returns a repo whose
    basename matches exactly (high confidence) — preferring the most-downloaded
    — or None. Best-effort and never raises, so an offline box just gets the
    original error."""
    try:
        from huggingface_hub import HfApi

        results = list(HfApi().list_models(search=model_name, sort="downloads", limit=25))
    except Exception:
        return None
    target = model_name.split("/")[-1].lower()
    for m in results:  # already sorted by downloads, descending
        if m.id.split("/")[-1].lower() == target:
            return m.id
    return None


def _assert_gpu_arch_supported(device: str) -> None:
    """Fail fast, with an actionable message, when the GPU's compute capability
    is not in this PyTorch build's kernel list — the "no kernel image is
    available for execution on the device" failure (e.g. a Blackwell B200 /
    sm_100 on a torch compiled only up to sm_90).

    Crucially this is NOT a model-size problem: every CUDA kernel fails, so
    switching to a smaller model does not help and is not a valid spec
    revision. The fix is a Blackwell-capable PyTorch."""
    if device != "cuda" or not torch.cuda.is_available():
        return
    try:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        arches = list(torch.cuda.get_arch_list())
        name = torch.cuda.get_device_name(0)
    except Exception:
        return
    if not arches or sm in arches:
        return
    raise RuntimeError(
        f"GPU not supported by this PyTorch build: {name} is {sm}, but torch "
        f"{torch.__version__} (CUDA {torch.version.cuda}) only ships kernels for "
        f"{arches}. No CUDA computation can run on this device — a SMALLER MODEL "
        f"WILL NOT HELP and is not a valid spec revision. Fix the environment: "
        f"install a {sm}-capable PyTorch (for Blackwell/B200: `pip install "
        f"--upgrade torch --index-url https://download.pytorch.org/whl/cu128`), "
        f"or pass device='cpu' for a tiny pilot only."
    )


def _looks_like_missing_repo(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"RepositoryNotFoundError", "HFValidationError", "EntryNotFoundError"}:
        return True
    text = str(exc).lower()
    return (
        "404" in text
        or "not a valid model identifier" in text
        or "is not a local folder" in text
    )


def infer_model_type(model_name: str) -> str:
    lower = model_name.lower()
    if "llama" in lower:
        return "llama"
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    if "mistral" in lower:
        return "mistral"
    if "deepseek" in lower:
        return "deepseek"
    return "unknown"


def make_quantization_config(mode: Optional[str]) -> Optional[BitsAndBytesConfig]:
    if mode is None:
        return None
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True)
    raise ValueError(f"Unknown quantization mode: {mode}")


@dataclass
class ModelHandle:
    model_name: str
    model_id: str
    model: Any
    tokenizer: Any
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    model_type: str = "unknown"

    def _decoder_layers(self) -> Any:
        """The decoder block ModuleList, across HF architectures.

        Handles GPT-2 / GPT-J / GPT-Neo (`transformer.h`), Llama / Qwen / Mistral
        (`model.layers`), GPTNeoX / Pythia (`gpt_neox.layers`), and the
        double-nested / multimodal variants — so head/activation patching works on
        whatever open-weights model the agent picks, not just gpt2.
        """
        m = self.model
        inner = getattr(m, "model", None)
        candidates = [
            getattr(inner, "layers", None),                                   # Llama/Qwen/Mistral
            getattr(getattr(m, "gpt_neox", None), "layers", None),            # GPTNeoX / Pythia
            getattr(getattr(m, "transformer", None), "h", None),              # GPT-2 / GPT-J / Neo
            getattr(getattr(inner, "language_model", None), "layers", None),  # multimodal
            getattr(getattr(inner, "model", None), "layers", None),           # double-nested
        ]
        for layers in candidates:
            if layers is not None:
                return layers
        raise ValueError(
            "Could not locate the decoder layers on this model "
            f"({type(self.model).__name__}); architecture not recognized."
        )

    @property
    def n_layers(self) -> int:
        try:
            return len(self._decoder_layers())
        except ValueError:
            pass
        config = self.model.config
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(config, name):
                return int(getattr(config, name))
        if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
            return int(config.text_config.num_hidden_layers)
        raise ValueError("Could not infer number of layers")

    @property
    def d_model(self) -> int:
        config = self.model.config
        for name in ("hidden_size", "d_model", "dim", "n_embd"):
            if hasattr(config, name):
                return int(getattr(config, name))
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        raise ValueError("Could not infer hidden size")

    @property
    def n_heads(self) -> int:
        config = self.model.config
        for name in ("num_attention_heads", "n_head", "num_heads"):
            if hasattr(config, name):
                return int(getattr(config, name))
        if hasattr(config, "text_config") and hasattr(config.text_config, "num_attention_heads"):
            return int(config.text_config.num_attention_heads)
        raise ValueError("Could not infer number of attention heads")

    def input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def layer(self, layer_idx: int) -> Any:
        if layer_idx < 0:
            layer_idx = self.n_layers + layer_idx
        return self._decoder_layers()[layer_idx]

    def filter_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self.model_type in MODELS_WITHOUT_SYSTEM_ROLE:
            return [m for m in messages if m.get("role") != "system"]
        return messages

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        messages = self.filter_messages(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        parts = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n\n".join(parts)

    def tokenize(self, text: str, padding: bool = False) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            add_special_tokens=False,
        ).to(self.input_device())

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        **generation_kwargs: Any,
    ) -> str:
        inputs = self.tokenize(prompt)
        input_length = inputs["input_ids"].shape[1]
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs.update({"do_sample": True, "temperature": temperature})
            kwargs.update(generation_kwargs)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)
        new_tokens = output_ids[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        **generation_kwargs: Any,
    ) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.input_device())
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs.update({"do_sample": True, "temperature": temperature})
            kwargs.update(generation_kwargs)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)
        outputs = []
        for row, input_length in enumerate(input_lengths):
            new_tokens = output_ids[row][input_length:]
            outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return outputs

    def cleanup(self) -> None:
        if hasattr(self.model, "cpu"):
            self.model.cpu()
        torch.cuda.empty_cache()


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    quantization: Optional[str] = None,
    trust_remote_code: bool = True,
) -> ModelHandle:
    _assert_gpu_arch_supported(device)
    model_id = resolve_model_id(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        # Dynamic fallback: the name may be a colloquial id (e.g. 'pythia-160m')
        # that lives under an org prefix on the Hub. Search for the canonical
        # repo and retry, rather than failing on a stale alias table.
        found = _resolve_via_hub_search(model_name) if _looks_like_missing_repo(exc) else None
        if not found or found == model_id:
            raise
        model_id = found
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = make_quantization_config(quantization)
    load_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_id,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto" if device == "cuda" else None,
        "attn_implementation": "eager",
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    elif model_name not in PREQUANTIZED_ALIASES:
        load_kwargs["dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    except TypeError:
        if "dtype" in load_kwargs:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return ModelHandle(
        model_name=model_name,
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_type=infer_model_type(model_name),
    )
