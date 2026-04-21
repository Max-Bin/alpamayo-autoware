# SPDX-License-Identifier: Apache-2.0
"""FlashDrive orchestrator. ``apply_flashdrive(model, config)`` patches in
each enabled optimisation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class FlashDriveConfig:
    """Toggle individual FlashDrive optimisations."""

    # Cache the velocity field at middle denoising steps; only recompute
    # at the first + last. ~30 ms saved on a 5-step diffusion run.
    adaptive_flow: bool = False
    adaptive_recompute_indices: tuple[int, ...] | None = None

    # Pack q/k/v projections into one Linear; gate/up MLP into another.
    kernel_fusion_qkv: bool = False
    kernel_fusion_mlp: bool = False

    # Cache ViT outputs across temporal frames (3-of-4 overlap → 75% reuse).
    streaming_vision: bool = False
    streaming_vision_window: int = 32

    # LM-half streaming: pre-RoPE K/V + hidden_out cache per decoder layer
    # for cached frame token ranges. Lossy approximation per FlashDrive
    # paper (action expert needs fine-tune to absorb drift).
    streaming_lm: bool = False
    streaming_lm_window: int = 128

    @classmethod
    def from_strings(cls, options: Iterable[str]) -> "FlashDriveConfig":
        cfg = cls()
        for opt in options:
            opt = opt.strip().lower()
            if not opt:
                continue
            if not hasattr(cfg, opt):
                raise ValueError(
                    f"unknown flashdrive option: {opt!r} "
                    f"(known: {[f.name for f in cfg.__dataclass_fields__.values()]})"
                )
            setattr(cfg, opt, True)
        return cfg


def apply_flashdrive(model: torch.nn.Module, config: FlashDriveConfig) -> None:
    """Patch ``model`` in place with the optimisations selected in ``config``."""
    # cuDNN attention is ~25% faster than default SDPA at decode shape on sm_120.
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:  # noqa: BLE001
        pass

    enabled: list[str] = []

    if config.adaptive_flow:
        from alpamayo1_5.flashdrive import adaptive_flow
        adaptive_flow.install(model.diffusion, recompute_indices=config.adaptive_recompute_indices)
        enabled.append("adaptive_flow")

    if config.kernel_fusion_qkv or config.kernel_fusion_mlp:
        from alpamayo1_5.flashdrive import kernel_fusion
        kernel_fusion.install(
            model.vlm,
            fuse_qkv=config.kernel_fusion_qkv,
            fuse_mlp=config.kernel_fusion_mlp,
        )
        enabled.append(f"kernel_fusion(qkv={config.kernel_fusion_qkv},mlp={config.kernel_fusion_mlp})")

    # streaming_lm must install BEFORE streaming_vision: the vision cache
    # publishes per-image layouts to the LM coordinator.
    if config.streaming_lm:
        from alpamayo1_5.flashdrive import streaming_lm
        streaming_lm.install(model.vlm, max_entries_per_layer=config.streaming_lm_window)
        enabled.append("streaming_lm")

    if config.streaming_vision:
        from alpamayo1_5.flashdrive import streaming_vision
        streaming_vision.install(model.vlm, window=config.streaming_vision_window)
        enabled.append("streaming_vision")

    LOGGER.info("FlashDrive enabled: %s", enabled or "(none — pure baseline)")
