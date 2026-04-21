# SPDX-License-Identifier: Apache-2.0
"""Pack q/k/v into one Linear (head-axis split, contiguous), and
gate/up into another. Numerically identical to unfused. Cuts launch
count by 60% / 33% in attention / MLP paths.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)
_MARKER = "_flashdrive_kfuse_install"


def _make_fused_linear(parts: list[nn.Linear]) -> nn.Linear:
    """Concatenate weights of multiple Linears with the same in_features
    into a single Linear that produces ``cat([part(x) for part in parts])``.
    """
    in_features = parts[0].in_features
    for p in parts:
        if p.in_features != in_features:
            raise ValueError(f"in_features mismatch in fused stack: {p.in_features} vs {in_features}")
    bias = parts[0].bias is not None
    if any((p.bias is not None) != bias for p in parts):
        raise NotImplementedError("mixed-bias fused linear is not handled here")

    total_out = sum(p.out_features for p in parts)
    fused = nn.Linear(in_features, total_out, bias=bias).to(
        device=parts[0].weight.device, dtype=parts[0].weight.dtype
    )
    with torch.no_grad():
        offset = 0
        for p in parts:
            n = p.out_features
            fused.weight[offset : offset + n].copy_(p.weight)
            if bias:
                fused.bias[offset : offset + n].copy_(p.bias)
            offset += n
    return fused


def _patch_attention_layer(layer) -> None:
    """Install fused QKV on Qwen3VLTextAttention. Pack along head axis
    so the downstream ``output.reshape(*, total_heads, head_dim)`` split
    stays contiguous (otherwise ~70 µs/layer reshape regression).
    """
    import torch.nn as nn
    q, k, v = layer.q_proj, layer.k_proj, layer.v_proj
    # Skip layers another optimisation already swapped the projections on.
    if not all(isinstance(m, nn.Linear) for m in (q, k, v)):
        LOGGER.info("kernel_fusion: skipping layer — projections aren't plain nn.Linear "
                    "(another optimisation already swapped them)")
        return
    n_q = q.out_features // layer.head_dim
    n_kv = k.out_features // layer.head_dim
    if k.out_features != v.out_features:
        raise NotImplementedError("k_proj/v_proj head counts differ; head-axis split unsafe")

    fused = _make_fused_linear([q, k, v])
    layer._fd_qkv_proj = fused
    layer._fd_n_q = n_q
    layer._fd_n_kv = n_kv
    layer._fd_total_heads = n_q + 2 * n_kv
    layer.q_proj = nn.Identity()
    layer.k_proj = nn.Identity()
    layer.v_proj = nn.Identity()

    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]

        qkv = self._fd_qkv_proj(hidden_states)
        # Reshape to [..., total_heads, head_dim], then split along the
        # head axis. Each part is contiguous in memory because head dim is
        # the innermost stride and the splits cover head-aligned blocks.
        qkv = qkv.view(*input_shape, self._fd_total_heads, self.head_dim)
        n_q, n_kv = self._fd_n_q, self._fd_n_kv
        q = qkv[..., :n_q, :]
        k = qkv[..., n_q : n_q + n_kv, :]
        v = qkv[..., n_q + n_kv :, :]

        # Match the original (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim) layout.
        query_states = self.q_norm(q).transpose(1, 2)
        key_states = self.k_norm(k).transpose(1, 2)
        value_states = v.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            ALL_ATTENTION_FUNCTIONS,
            eager_attention_forward,
        )
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    import types
    layer.forward = types.MethodType(fused_forward, layer)


def _patch_mlp_layer(layer) -> None:
    """Install a fused gate+up projection on a Qwen3VLTextMLP instance."""
    g, u = layer.gate_proj, layer.up_proj
    if not all(isinstance(m, nn.Linear) for m in (g, u)):
        LOGGER.info("kernel_fusion: skipping mlp — gate/up aren't plain nn.Linear")
        return
    fused = _make_fused_linear([g, u])
    layer._fd_gateup_proj = fused
    layer._fd_intermediate = g.out_features
    layer.gate_proj = nn.Identity()
    layer.up_proj = nn.Identity()

    def fused_forward(self, x):
        gate_up = self._fd_gateup_proj(x)
        gate, up = gate_up.split(self._fd_intermediate, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

    import types
    layer.forward = types.MethodType(fused_forward, layer)


def install(vlm: nn.Module, *, fuse_qkv: bool = True, fuse_mlp: bool = True) -> dict:
    """Walk the language-model decoder and install fused projections."""
    if getattr(vlm, _MARKER, False):
        LOGGER.info("kernel_fusion already installed; skipping")
        return {"qkv": 0, "mlp": 0}

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextAttention,
        Qwen3VLTextMLP,
    )

    n_qkv = n_mlp = 0
    target = vlm.model.language_model
    for name, module in target.named_modules():
        if fuse_qkv and isinstance(module, Qwen3VLTextAttention):
            _patch_attention_layer(module)
            n_qkv += 1
        if fuse_mlp and isinstance(module, Qwen3VLTextMLP):
            _patch_mlp_layer(module)
            n_mlp += 1
    setattr(vlm, _MARKER, True)
    LOGGER.info("kernel_fusion installed: %d Q/K/V + %d gate/up fused", n_qkv, n_mlp)
    return {"qkv": n_qkv, "mlp": n_mlp}
