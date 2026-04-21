# SPDX-License-Identifier: Apache-2.0
"""W4A8 PARO quantisation via vLLM's AWQ-Marlin kernel (sm_120 native
with has_zp=True). Reads ``FlashDriveVLA/Alpamayo-1.5-10B-finetuned-PARO``
(``paroquant_marlin_w4a8`` format, 252 quantized modules).

Hybrid dispatch:
  M ≥ 16 (prefill): Marlin W4A8 GEMM, ~1.51× faster than BF16 at M=3000.
  M < 16 (decode):  cached dequantized weight + plain matmul. Marlin's
                    ~36 µs per-call overhead dominates at M=1; cached-
                    BF16 path matches baseline GEMV (~17 µs).

Combined with the rest of the FlashDrive stack contributes +0.13× e2e
(2.02× → 2.16× steady-state on T4 streaming bench).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)
_MARKER = "_flashdrive_paro_install"


_PARO_QSUFFIX = ".rotate_linear.qlinear"  # qweight, qzeros, scales, g_idx, ...
_PARO_RSUFFIX = ".rotate_linear.rotation"  # theta, pairs, channel_scales


def _quantized_module_names(ckpt_path: str) -> set[str]:
    """Return the set of underlying ``nn.Linear`` submodule names whose
    weights are PARO-quantized in the checkpoint. Strips the
    ``.rotate_linear.qlinear`` suffix that PARO adds at save time.
    """
    index_file = Path(ckpt_path) / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    out = set()
    for k in weight_map:
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            if base.endswith(_PARO_QSUFFIX):
                base = base[: -len(_PARO_QSUFFIX)]
            out.add(base)
    return out


def _flatten_paro_state(state: dict) -> dict:
    """Strip the ``.rotate_linear.qlinear`` and ``.rotate_linear.rotation``
    suffixes so PARO state-dict keys load directly into our flat
    ``MarlinRotatedLinear`` modules.
    """
    out = {}
    for k, v in state.items():
        nk = k.replace(_PARO_QSUFFIX + ".", ".").replace(_PARO_RSUFFIX + ".", ".")
        out[nk] = v
    return out


class MarlinRotatedLinear(nn.Module):
    """ParoQuant W4A8 inference Linear backed by vLLM's Marlin kernel.

    Stores the same flat parameter set the PARO checkpoint serializes
    (``qweight``, ``qzeros``, ``scales``, ``g_idx``, ``g_idx_sort_indices``,
    ``input_global_scale``, plus the rotation buffers ``theta``, ``pairs``,
    ``channel_scales``) so a key-rewriting state dict load works directly.

    Forward:
      1. Pairwise Givens rotation (``torch.ops.rotation.rotate``).
      2. INT8-activation Marlin W4A8 GEMM (``apply_awq_marlin_linear``)
         with ``input_dtype=torch.int8`` so input gets quantised inside
         the kernel and the INT8 tensor cores (sm_120 native) light up.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        group_size: int = 128,
        bits: int = 4,
        krot: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits

        # Marlin packed buffers — shapes verified against the published
        # paroquant_marlin_w4a8 checkpoint:
        #   qweight  (in_features // 16, out_features * 2) int32  — 4-bit
        #            weights packed two per int8, then Marlin tiled.
        #   qzeros   (in_features // group_size, out_features // 8) int32
        #   scales   (in_features // group_size, out_features) float16
        #   g_idx, g_idx_sort_indices: empty for static-group (the AWQ
        #            path doesn't use g_idx; Marlin still wants the
        #            tensors to exist for its kernel signature).
        #   workspace: (sm_count,) int32 — set after load via
        #            ``marlin_make_workspace_new``.
        #   input_global_scale: scalar float32 — fp8/int8 act calibration.
        n_groups = in_features // group_size
        self.register_buffer("qweight", torch.zeros(in_features // 16, out_features * 2, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(n_groups, out_features // 8, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(n_groups, out_features, dtype=torch.float16))
        self.register_buffer("g_idx", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("g_idx_sort_indices", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("workspace", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("input_global_scale", torch.zeros((), dtype=torch.float32))

        # Rotation buffers (same as upstream RotateQuantizedLinear).
        self.register_buffer("theta", torch.empty(krot, in_features // 2, dtype=torch.float16))
        self.register_buffer("pairs", torch.empty(krot, in_features, dtype=torch.int16))
        self.register_buffer("channel_scales", torch.empty(1, in_features, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        # Hybrid kernel dispatch by M:
        #   - prefill (M >= 16): Marlin W4A8 GEMM. Wins 1.34-1.51x at M>=1024.
        #     Marlin requires fp16; we cast in/out around it.
        #   - decode  (M <  16): cached dequantized weight @ matmul. Marlin
        #     and custom Triton W4 GEMV both lose to plain GEMV at M=1 on
        #     sm_120 because plain BF16 is already at HBM bandwidth (~18 µs)
        #     and the W4 launch overhead (36-200 µs) dominates. Stay in
        #     the model's working dtype (BF16) end-to-end here so we
        #     don't pay cast overhead per call (~7 µs × 252 Linears × 17
        #     decode steps ≈ 30 ms otherwise).
        flat_m = x.shape[:-1].numel()
        if flat_m >= 16:
            x_f16 = x.to(torch.float16) if in_dtype != torch.float16 else x
            x_f16 = torch.ops.rotation.rotate(x_f16, self.pairs, self.theta, self.channel_scales)
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                apply_awq_marlin_linear,
            )
            from vllm.scalar_type import scalar_types
            out = apply_awq_marlin_linear(
                input=x_f16,
                weight=self.qweight,
                weight_scale=self.scales,
                weight_zp=self.qzeros,
                g_idx=self.g_idx,
                g_idx_sort_indices=self.g_idx_sort_indices,
                workspace=self.workspace,
                quant_type=scalar_types.uint4,
                output_size_per_partition=self.out_features,
                input_size_per_partition=self.in_features,
                input_global_scale=(
                    self.input_global_scale if self.input_global_scale.numel() > 0 else None
                ),
                bias=self.bias,
                input_dtype=torch.int8,
            )
            if out.dtype != in_dtype:
                out = out.to(in_dtype)
            return out

        # Decode path — keep BF16 (or whatever the input dtype is) end-to-end.
        if not hasattr(self, "_w_native_t"):
            w_in_out_fp16 = self._dequant_marlin_to_fp16()
            self._w_native_t = w_in_out_fp16.t().contiguous().to(in_dtype)
        # Rotate in BF16 (paroquant rotation kernel supports it natively).
        x_rot = torch.ops.rotation.rotate(
            x,
            self.pairs,
            self.theta.to(in_dtype),
            self.channel_scales.to(in_dtype),
        )
        out = x_rot @ self._w_native_t.t()
        if self.bias is not None:
            out = out + self.bias.to(in_dtype)
        return out

    @torch.no_grad()
    def _dequant_marlin_to_fp16(self) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_awq_marlin_linear,
        )
        from vllm.scalar_type import scalar_types
        device = self.qweight.device
        eye = torch.eye(self.in_features, device=device, dtype=torch.float16)
        # apply_awq_marlin_linear(eye, W) returns (in, out) — that IS the dequantized weight.
        return apply_awq_marlin_linear(
            input=eye, weight=self.qweight, weight_scale=self.scales,
            weight_zp=self.qzeros, g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices, workspace=self.workspace,
            quant_type=scalar_types.uint4,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            input_global_scale=(
                self.input_global_scale if self.input_global_scale.numel() > 0 else None
            ),
            input_dtype=torch.int8,
        )



def install(
    model: nn.Module,
    *,
    ckpt_path: Optional[str] = None,
    bits: int = 4,
    group_size: int = 128,
    krot: int = 8,
) -> int:
    """Replace ``nn.Linear`` modules listed in the PARO checkpoint's
    weight map with :class:`MarlinRotatedLinear` and load the W4A8
    Marlin-packed weights.

    ``model`` is the Alpamayo1_5 instance returned by
    ``alpamayo1_5.flashdrive.load_helpers``. ``ckpt_path`` points at the
    ``FlashDriveVLA/Alpamayo-1.5-10B-finetuned-PARO`` directory locally.
    """
    if ckpt_path is None:
        raise ValueError("paro_w4a8 requires ckpt_path (e.g. models/Alpamayo-1.5-10B-finetuned-PARO)")
    if getattr(model, _MARKER, False):
        LOGGER.info("paro_w4a8 already installed")
        return 0

    # paroquant ships the rotation kernel as torch.ops.rotation.rotate;
    # importing the package registers it. AutoAWQ activation alias is no
    # longer needed (we don't go through the awq path) but still cheap.
    import transformers.activations as _act
    if not hasattr(_act, "PytorchGELUTanh"):
        _act.PytorchGELUTanh = _act.GELUActivation
    import paroquant.kernels.cuda  # noqa: F401  registers torch.ops.rotation.rotate

    quantized_names = _quantized_module_names(ckpt_path)
    LOGGER.info("PARO checkpoint declares %d quantized linears", len(quantized_names))

    device = next(model.parameters()).device
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name not in quantized_names:
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        new = MarlinRotatedLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            group_size=group_size,
            bits=bits,
            krot=krot,
        ).to(device)
        setattr(parent, attr, new)
        swapped += 1

    if swapped != len(quantized_names):
        LOGGER.warning(
            "PARO swap mismatch: %d swapped vs %d declared in checkpoint",
            swapped, len(quantized_names),
        )

    from safetensors.torch import load_file
    state: dict[str, torch.Tensor] = {}
    for sf in sorted(Path(ckpt_path).glob("*.safetensors")):
        state.update(load_file(str(sf)))

    # Drop the patch_embed weight from the PARO state — that key has the
    # Cosmos flat (1152, 1536) layout and would clash with our reshaped
    # Conv3d (1152, 3, 2, 16, 16); the BF16 shard already loaded the
    # correct one.
    state.pop("vlm.model.visual.patch_embed.proj.weight", None)

    state = _flatten_paro_state(state)

    # PyTorch's load_state_dict(strict=False, assign=True) still rejects
    # shape mismatches on buffers (workspace 0→188, input_global_scale
    # ()→() but flagged anyway). Manually assign each loaded tensor to
    # its destination buffer, replacing whatever placeholder we set in
    # the constructor. This is the simplest correct path for a heterogenous
    # buffer-shape load.
    own_state = dict(model.named_buffers(recurse=True))
    own_params = dict(model.named_parameters(recurse=True))
    n_loaded = 0
    n_skipped = 0
    for k, v in state.items():
        v = v.to(device)
        if k in own_state:
            # Replace buffer in-place by reassigning on the parent.
            parent_name, attr = k.rsplit(".", 1) if "." in k else ("", k)
            parent = model.get_submodule(parent_name) if parent_name else model
            try:
                delattr(parent, attr)
            except AttributeError:
                pass
            parent.register_buffer(attr, v)
            n_loaded += 1
        elif k in own_params:
            with torch.no_grad():
                own_params[k].data = v.to(own_params[k].dtype)
            n_loaded += 1
        else:
            n_skipped += 1
    LOGGER.info("PARO load: assigned %d tensors, skipped %d unexpected", n_loaded, n_skipped)

    setattr(model, _MARKER, True)
    LOGGER.info("paro_w4a8 installed: %d Linears swapped (Marlin W4A8 backend)", swapped)
    return swapped
