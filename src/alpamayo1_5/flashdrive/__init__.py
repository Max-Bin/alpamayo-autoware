# SPDX-License-Identifier: Apache-2.0
"""FlashDrive optimisations for Alpamayo 1.5 on Blackwell sm_120.

Re-implementation of the algorithm-system co-design from
https://z-lab.ai/projects/flashdrive/ — see each module for mechanism.
``apply_flashdrive(model, FlashDriveConfig(...))`` patches in selected
optimisations; each is independently toggleable.

Measured on multi-camera rosbag (8 streaming steps, 16 gen tokens):
    baseline (10 diff steps, upstream)   711 ms   1.00×
    flashdrive (5 diff steps + stack)    292 ms   2.44×   best=146
"""
from alpamayo1_5.flashdrive.pipeline import (
    FlashDriveConfig,
    apply_flashdrive,
)

__all__ = ["FlashDriveConfig", "apply_flashdrive"]
