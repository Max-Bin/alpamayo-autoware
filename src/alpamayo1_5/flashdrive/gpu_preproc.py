# SPDX-License-Identifier: Apache-2.0
"""FlashDrive optimisation #10: GPU-side JPEG decode + resize.

Background
----------
Default ROS pipeline for compressed camera frames goes:

  CompressedImage msg → cv2.imdecode (CPU) → cv2.cvtColor BGR→RGB (CPU)
  → cv2.resize if needed (CPU) → torch.from_numpy → CPU→GPU transfer

For 4 cams × 4 frames at 1080×1920, that's ~150 ms per callback on
the critical path — all wasted CPU work. The JPEGs can instead be
decoded directly on GPU with ``torchvision.io.decode_jpeg(device="cuda")``
and resized via ``F.interpolate`` in one fused GPU pass.

Usage
-----
1. ROS image callback: stash raw JPEG bytes as ``torch.uint8`` (no
   decode here).

       jpeg_bytes = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
       camera_buffers[topic].append((stamp, jpeg_bytes))

2. At inference time: hand the list of JPEG byte tensors to
   ``decode_jpeg_batch_gpu`` to get the (N, 3, H, W) uint8 tensor on
   GPU at the model's native 560×1008 resolution.

       imgs = decode_jpeg_batch_gpu(jpeg_buffers, target_hw=(560, 1008))

The same helper is useful for bench scripts that read rosbags — moves
the per-frame decode from CPU to GPU.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch

LOGGER = logging.getLogger(__name__)


def decode_jpeg_batch_gpu(
    jpeg_buffers: Sequence[torch.Tensor],
    *,
    target_hw: tuple[int, int] = (560, 1008),
    device: str = "cuda",
) -> torch.Tensor:
    """Batch-decode JPEG bytes on GPU and resize to ``target_hw``.

    Args:
        jpeg_buffers: list of ``torch.uint8`` 1-D tensors, each holding
            one JPEG file's raw bytes (CompressedImage.data).
        target_hw: (H, W) of the model's native input. Frames decoded
            at native size are resized via bicubic ``F.interpolate``.
        device: target CUDA device.

    Returns:
        ``torch.uint8`` tensor of shape ``(N, 3, H, W)`` on device.
    """
    import torchvision

    decoded = [torchvision.io.decode_jpeg(buf, device=device) for buf in jpeg_buffers]
    stacked = torch.stack(decoded)  # (N, 3, H, W) uint8 on GPU
    if stacked.shape[-2:] != target_hw:
        stacked = (
            torch.nn.functional.interpolate(
                stacked.float(), size=target_hw, mode="bicubic", align_corners=False,
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )
    return stacked
