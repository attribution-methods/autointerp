"""Model loading, chat formatting, and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ALIASES: Dict[str, str] = {
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


def resolve_model_id(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


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
    dtype: torch.dtype = torch.bfloat16
    model_type: str = "unknown"

    @property
    def n_layers(self) -> int:
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
                return len(inner.language_model.layers)
            if hasattr(inner, "layers"):
                return len(inner.layers)
            if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                return len(inner.model.layers)
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
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
                return inner.language_model.layers[layer_idx]
            if hasattr(inner, "layers"):
                return inner.layers[layer_idx]
            if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                return inner.model.layers[layer_idx]
        raise ValueError(f"Could not access layer {layer_idx}")

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
    dtype: torch.dtype = torch.bfloat16,
    quantization: Optional[str] = None,
    trust_remote_code: bool = True,
) -> ModelHandle:
    model_id = resolve_model_id(model_name)
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
