# SPDX-License-Identifier: Apache-2.0
"""Adaptive-step flow matching: cache the velocity field at middle
denoising steps, recompute only at first + last (or caller-specified
set). Velocity ``||v_{i+1} - v_i|| / ||v_i||`` profile is a U-shape
(~27% at edges, < 6% through middle), so middle reuse is ~lossless
within ~0.08 m ADE on Alpamayo 1.5. Cuts 5-step diffusion to 2 calls.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

from alpamayo1_5.diffusion.flow_matching import FlowMatching

LOGGER = logging.getLogger(__name__)
_MARKER = "_flashdrive_adaptive_install"


def install(
    diffusion: FlowMatching,
    *,
    recompute_indices: Optional[Iterable[int]] = None,
) -> None:
    """Replace ``diffusion._euler`` with an adaptive-step variant in place.

    Args:
        diffusion: the loaded ``FlowMatching`` instance.
        recompute_indices: which step indices actually call ``step_fn``.
            ``None`` defaults to ``{0, num_inference_steps - 1}``. Pass an
            explicit set if you want intermediate refresh (e.g. ``{0, 2, 4}``
            for a 5-step run).
    """
    if not isinstance(diffusion, FlowMatching):
        raise TypeError(
            f"adaptive_flow expects a FlowMatching instance, got {type(diffusion).__name__}"
        )
    if getattr(diffusion, _MARKER, False):
        LOGGER.info("adaptive_flow already installed; refreshing recompute_indices")
    diffusion._fd_recompute_indices = (
        tuple(recompute_indices) if recompute_indices is not None else None
    )

    original_euler = diffusion._euler

    @torch.no_grad()
    def _euler_adaptive(
        batch_size,
        step_fn,
        unguided_step_fn=None,
        device=torch.device("cpu"),
        return_all_steps=False,
        inference_step=None,
        inference_guidance_weight=None,
        use_classifier_free_guidance=None,
        temperature=1.0,
    ):
        x = torch.randn(batch_size, *diffusion.x_dims, device=device) * temperature
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(diffusion.x_dims)
        if return_all_steps:
            all_steps = [x]

        recompute = (
            set(diffusion._fd_recompute_indices)
            if diffusion._fd_recompute_indices is not None
            else {0, inference_step - 1}
        )

        v_cached: torch.Tensor | None = None
        for i in range(inference_step):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            if i in recompute or v_cached is None:
                if use_classifier_free_guidance:
                    v = diffusion._guided_v(
                        step_fn=step_fn,
                        x=x,
                        t=t_start,
                        unguided_step_fn=unguided_step_fn,
                        inference_guidance_weight=inference_guidance_weight,
                    )
                else:
                    v = step_fn(x=x, t=t_start)
                v_cached = v
            else:
                v = v_cached
            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)
        if return_all_steps:
            return torch.stack(all_steps, dim=1), time_steps
        return x

    diffusion._euler = _euler_adaptive
    diffusion._fd_original_euler = original_euler
    setattr(diffusion, _MARKER, True)
    LOGGER.info(
        "adaptive_flow installed (recompute_indices=%s)",
        diffusion._fd_recompute_indices or "{0, last}",
    )


def uninstall(diffusion: FlowMatching) -> None:
    """Restore the original Euler integrator. Idempotent."""
    if not getattr(diffusion, _MARKER, False):
        return
    diffusion._euler = diffusion._fd_original_euler
    del diffusion._fd_original_euler
    del diffusion._fd_recompute_indices
    setattr(diffusion, _MARKER, False)
