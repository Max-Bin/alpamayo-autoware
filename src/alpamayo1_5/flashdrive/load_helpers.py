# SPDX-License-Identifier: Apache-2.0
"""Loader shims so we don't need access to gated nvidia/Cosmos-Reason2-8B.

The published ``FlashDriveVLA/Alpamayo-1.5-10B-finetuned`` config points at
``nvidia/Cosmos-Reason2-8B`` for the architecture template; that repo is
gated. The public ``Qwen/Qwen3-VL-8B-Instruct`` is architecturally close
but uses a Conv3d patch embedding (shape ``[1152, 3, 2, 16, 16]``) where
Cosmos used a Linear (shape ``[1152, 1536]``). The two are
element-identical — same 1152×1536=1152×3×2×16×16 — and reshape produces
a valid Conv3d weight.

``patch_alpamayo15_for_qwen3vl(model_dir)`` rewrites the ``config.json``
in place so ``vlm_name_or_path = "Qwen/Qwen3-VL-8B-Instruct"`` and
provides a state-dict pre-hook that reshapes the patch-embed weight on
the fly during ``from_pretrained``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)

_PATCH_EMBED_KEY = "vlm.model.visual.patch_embed.proj.weight"
_PATCH_EMBED_TARGET_SHAPE = (1152, 3, 2, 16, 16)
_PATCH_EMBED_FLAT_SHAPE = (1152, 1536)


def patch_alpamayo15_config_for_qwen3vl(model_dir: str | Path) -> None:
    """Rewrite ``config.json`` in place to swap the gated Cosmos-Reason2-8B
    reference for the public Qwen3-VL-8B-Instruct one. Idempotent."""
    cfg_path = Path(model_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("vlm_name_or_path") == "Qwen/Qwen3-VL-8B-Instruct":
        return
    if cfg.get("vlm_name_or_path") not in ("nvidia/Cosmos-Reason2-8B", "Qwen/Qwen3-VL-8B-Instruct"):
        LOGGER.warning(
            "config.json vlm_name_or_path is %r — leaving untouched",
            cfg.get("vlm_name_or_path"),
        )
        return
    cfg["vlm_name_or_path"] = "Qwen/Qwen3-VL-8B-Instruct"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    LOGGER.info("patched %s: vlm_name_or_path -> Qwen/Qwen3-VL-8B-Instruct", cfg_path)


def reshape_patch_embed_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reshape the flat 1152×1536 patch-embed weight to the Conv3d 5-D
    layout. Pure function — does not mutate ``state_dict``."""
    out = dict(state_dict)
    w = out.get(_PATCH_EMBED_KEY)
    if w is not None and tuple(w.shape) == _PATCH_EMBED_FLAT_SHAPE:
        out[_PATCH_EMBED_KEY] = w.view(*_PATCH_EMBED_TARGET_SHAPE)
        LOGGER.info(
            "reshaped %s: %s -> %s",
            _PATCH_EMBED_KEY, _PATCH_EMBED_FLAT_SHAPE, _PATCH_EMBED_TARGET_SHAPE,
        )
    return out


def load_full_state_dict(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read every shard under ``model_dir`` and apply the patch-embed reshape."""
    from safetensors import safe_open

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(Path(model_dir).glob("model-*-of-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return reshape_patch_embed_state_dict(state)
