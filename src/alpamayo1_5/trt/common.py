from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def legacy_cache_from_prompt_cache(prompt_cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return a legacy tuple cache from a transformers cache object."""
    if hasattr(prompt_cache, "to_legacy_cache"):
        return prompt_cache.to_legacy_cache()
    if isinstance(prompt_cache, tuple):
        return prompt_cache
    # StaticCache: extract key_cache/value_cache lists
    if hasattr(prompt_cache, "key_cache") and hasattr(prompt_cache, "value_cache"):
        return tuple(
            (prompt_cache.key_cache[i], prompt_cache.value_cache[i])
            for i in range(len(prompt_cache.key_cache))
        )
    raise TypeError(f"Unsupported prompt cache type: {type(prompt_cache)!r}")


def flatten_past_key_values(
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> list[torch.Tensor]:
    """Flatten a per-layer KV cache tuple into an ordered tensor list."""
    flat: list[torch.Tensor] = []
    for past_key, past_value in past_key_values:
        flat.extend([past_key, past_value])
    return flat


def past_key_value_input_names(num_layers: int) -> list[str]:
    """Create stable ONNX input names for the KV cache."""
    names: list[str] = []
    for layer_idx in range(num_layers):
        names.append(f"past_key_{layer_idx:02d}")
        names.append(f"past_value_{layer_idx:02d}")
    return names


def build_dynamic_axes(num_layers: int) -> dict[str, dict[int, str]]:
    """Dynamic axes for a denoiser graph with dynamic batch and prompt length."""
    dynamic_axes: dict[str, dict[int, str]] = {
        "x": {0: "batch"},
        "t": {0: "batch"},
        "position_ids": {1: "batch"},
        "attention_mask": {0: "batch", 3: "total_seq_len"},
        "pred": {0: "batch"},
    }
    for layer_idx in range(num_layers):
        dynamic_axes[f"past_key_{layer_idx:02d}"] = {0: "batch", 2: "prompt_seq_len"}
        dynamic_axes[f"past_value_{layer_idx:02d}"] = {0: "batch", 2: "prompt_seq_len"}
    return dynamic_axes


def torch_dtype_to_onnxruntime_dtype(dtype: torch.dtype) -> type:
    """Map a torch dtype to the numpy dtype object expected by ORT I/O binding."""
    import numpy as np

    mapping = {
        torch.bool: np.bool_,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int64: np.int64,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported dtype for ONNX Runtime binding: {dtype}")
    return mapping[dtype]


def resolve_artifact_dir(path: str | Path) -> Path:
    """Expand and create an artifact directory."""
    artifact_dir = Path(path).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir
