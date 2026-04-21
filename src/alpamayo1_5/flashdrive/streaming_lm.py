# SPDX-License-Identifier: Apache-2.0
"""LM-half streaming inference: pre-RoPE K/V + hidden_out cache per
decoder layer for vision-frame token ranges.

Mechanism per prefill (when streaming layout is published):
  1. Partition prompt positions into NEW (cache miss) + OLD (cache hit).
  2. At each layer: project Q/K/V on NEW only; splice cached pre-RoPE
     K/V at OLD positions; rotary applied with current cos/sin so
     cached K re-rotates to its new MRoPE position; attention computes
     for NEW queries against full K/V; MLP runs on NEW only; assemble
     output (cached hidden_out at OLD, fresh at NEW).
  3. Cache pre-RoPE K, V, hidden_out per (layer_idx, image_hash) for
     the next call.

Accuracy: lossy approximation per FlashDrive paper — cached K/V/h were
produced under a different attention context. Paper recovers accuracy
by fine-tuning the action expert against the streaming approximation;
without that, trajectory ADE drifts ~0.1–0.45 m.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)
_MARKER = "_flashdrive_streaming_lm_install"


class _LMStreamCoordinator:
    """Per-call layout (set by streaming_vision) + per-layer
    (image_hash → (K_pre_rope, V, hidden_out)) caches.
    """

    def __init__(self, *, max_entries_per_layer: int = 16):
        self.max_entries = max_entries_per_layer
        self.layer_caches: dict[int, OrderedDict[bytes, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self.current_layout: list[tuple[bytes, int, int]] = []
        # Per-call decisions cached so all 36 layers share them.
        self.new_mask: Optional[torch.Tensor] = None
        self.new_idx: Optional[torch.Tensor] = None
        self.n_new: int = 0
        self.hit_layout: list[tuple[bytes, int, int]] = []
        self.miss_layout: list[tuple[bytes, int, int]] = []
        self.hits = 0
        self.misses = 0

    def set_layout(self, layout: list[tuple[bytes, int, int]]) -> None:
        self.current_layout = layout
        self.new_mask = None
        self.new_idx = None
        self.n_new = 0
        self.hit_layout = []
        self.miss_layout = []

    def get(self, layer_idx: int, image_hash: bytes):
        c = self.layer_caches.get(layer_idx)
        if c is None:
            return None
        out = c.get(image_hash)
        if out is not None:
            c.move_to_end(image_hash)
        return out

    def put(self, layer_idx: int, image_hash: bytes,
            k_pre_rope: torch.Tensor, v: torch.Tensor, hidden_out: torch.Tensor) -> None:
        c = self.layer_caches.setdefault(layer_idx, OrderedDict())
        c[image_hash] = (k_pre_rope, v, hidden_out)
        if len(c) > self.max_entries:
            c.popitem(last=False)


def _wrap_layer(layer: nn.Module, layer_idx: int, coord: _LMStreamCoordinator) -> None:
    """Replace ``layer.forward`` with the streaming version that skips
    MLP/o_proj/Q-K-V projection compute on cached token positions."""
    if getattr(layer, "_fd_streaming_lm_wrapped", False):
        return
    self_attn = layer.self_attn
    head_dim = self_attn.head_dim
    orig_forward = layer.forward

    def patched(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        layout = coord.current_layout
        # Only apply during prefill with a layout published; decode (T=1) → eager.
        if not layout or hidden_states.shape[1] == 1:
            return orig_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            ALL_ATTENTION_FUNCTIONS, eager_attention_forward, rotate_half,
        )

        T = hidden_states.shape[1]
        device = hidden_states.device

        # Compute hit/miss partition + new_idx ONCE per call (all 36 layers
        # share cache state because cold-start populates them together).
        if coord.new_mask is None or coord.new_mask.shape[0] != T:
            old_mask = torch.zeros(T, dtype=torch.bool, device=device)
            hit_layout, miss_layout = [], []
            for image_hash, p0, p1 in layout:
                if coord.get(layer_idx, image_hash) is not None:
                    old_mask[p0:p1] = True
                    hit_layout.append((image_hash, p0, p1))
                else:
                    miss_layout.append((image_hash, p0, p1))
            coord.new_mask = ~old_mask
            coord.new_idx = coord.new_mask.nonzero().squeeze(-1)
            coord.n_new = int(coord.new_idx.shape[0])
            coord.hit_layout = hit_layout
            coord.miss_layout = miss_layout
        new_idx = coord.new_idx
        n_new = coord.n_new
        n_old = T - n_new

        # Cold start (no hits at all) → run normal forward + derive K/V to cache.
        if n_old == 0:
            out = orig_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            with torch.no_grad():
                ln = layer.input_layernorm(hidden_states)
                input_shape = ln.shape[:-1]
                if hasattr(self_attn, "_fd_qkv_proj"):
                    qkv = self_attn._fd_qkv_proj(ln).view(*input_shape, self_attn._fd_total_heads, head_dim)
                    n_q, n_kv = self_attn._fd_n_q, self_attn._fd_n_kv
                    k_pre = self_attn.k_norm(qkv[..., n_q : n_q + n_kv, :]).transpose(1, 2)
                    v_pre = qkv[..., n_q + n_kv :, :].transpose(1, 2)
                else:
                    hs = (*input_shape, -1, head_dim)
                    k_pre = self_attn.k_norm(self_attn.k_proj(ln).view(hs)).transpose(1, 2)
                    v_pre = self_attn.v_proj(ln).view(hs).transpose(1, 2)
            for image_hash, p0, p1 in layout:
                coord.put(
                    layer_idx, image_hash,
                    k_pre[:, :, p0:p1, :].clone().detach(),
                    v_pre[:, :, p0:p1, :].clone().detach(),
                    out[:, p0:p1, :].clone().detach(),
                )
            coord.misses += 1
            return out

        # ---- streaming fast path ----
        residual_full = hidden_states
        ln_full = layer.input_layernorm(hidden_states)

        miss_ranges = list(coord.miss_layout)
        hit_ranges = []
        for image_hash, p0, p1 in coord.hit_layout:
            cached = coord.get(layer_idx, image_hash)
            if cached is not None and cached[0].shape[2] == (p1 - p0):
                hit_ranges.append((image_hash, p0, p1, cached))
            else:
                miss_ranges.append((image_hash, p0, p1))

        # Q/K/V projection ONLY on new positions (~25% of T at 75% overlap).
        ln_new = ln_full[:, new_idx, :]
        if hasattr(self_attn, "_fd_qkv_proj"):
            qkv_new = self_attn._fd_qkv_proj(ln_new).view(1, n_new, self_attn._fd_total_heads, head_dim)
            n_q, n_kv = self_attn._fd_n_q, self_attn._fd_n_kv
            q_new = self_attn.q_norm(qkv_new[..., :n_q, :]).transpose(1, 2)
            k_new = self_attn.k_norm(qkv_new[..., n_q : n_q + n_kv, :]).transpose(1, 2)
            v_new = qkv_new[..., n_q + n_kv :, :].transpose(1, 2)
        else:
            hs = (1, n_new, -1, head_dim)
            q_new = self_attn.q_norm(self_attn.q_proj(ln_new).view(hs)).transpose(1, 2)
            k_new = self_attn.k_norm(self_attn.k_proj(ln_new).view(hs)).transpose(1, 2)
            v_new = self_attn.v_proj(ln_new).view(hs).transpose(1, 2)

        # Assemble K_full/V_full at full sequence length.
        num_kv = self_attn._fd_n_kv if hasattr(self_attn, "_fd_n_kv") else (
            self_attn.k_proj.out_features // head_dim
        )
        k_full = torch.empty(1, num_kv, T, head_dim, device=ln_full.device, dtype=ln_full.dtype)
        v_full = torch.empty(1, num_kv, T, head_dim, device=ln_full.device, dtype=ln_full.dtype)
        k_full[:, :, new_idx, :] = k_new
        v_full[:, :, new_idx, :] = v_new
        for image_hash, p0, p1, (k_cached, v_cached, _) in hit_ranges:
            k_full[:, :, p0:p1, :] = k_cached
            v_full[:, :, p0:p1, :] = v_cached
            coord.hits += 1
        miss_k_v_per_range = [
            (h, p0, p1,
             k_full[:, :, p0:p1, :].detach().clone(),
             v_full[:, :, p0:p1, :].detach().clone())
            for h, p0, p1 in miss_ranges
        ]

        # Apply rotary: q at new positions only, k at full sequence (cached
        # K re-rotates to its current position via the new cos/sin on the fly).
        cos, sin = position_embeddings
        cos_new = cos[:, new_idx, :] if cos.dim() == 3 else cos
        sin_new = sin[:, new_idx, :] if sin.dim() == 3 else sin
        q_new = (q_new * cos_new.unsqueeze(1)) + (rotate_half(q_new) * sin_new.unsqueeze(1))
        k_full = (k_full * cos.unsqueeze(1)) + (rotate_half(k_full) * sin.unsqueeze(1))
        if past_key_values is not None:
            k_full, v_full = past_key_values.update(
                k_full, v_full, self_attn.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        attn_fn = eager_attention_forward
        if self_attn.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self_attn.config._attn_implementation]
        am_new = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.shape[-2] == T:
            am_new = attention_mask[:, :, new_idx, :]
        attn_out_new, _ = attn_fn(
            self_attn, q_new, k_full, v_full, am_new,
            dropout=0.0, scaling=self_attn.scaling, **kwargs,
        )
        attn_out_new = self_attn.o_proj(attn_out_new.reshape(1, n_new, -1).contiguous())
        hidden_after_attn_new = residual_full[:, new_idx, :] + attn_out_new

        # MLP on new only.
        ln2_new = layer.post_attention_layernorm(hidden_after_attn_new)
        hidden_out_new = hidden_after_attn_new + layer.mlp(ln2_new)

        # Assemble: cached hidden_out at hits, fresh at new.
        full_out = torch.empty_like(hidden_states)
        full_out[:, new_idx, :] = hidden_out_new.to(full_out.dtype)
        for _, p0, p1, (_, _, h_cached) in hit_ranges:
            full_out[:, p0:p1, :] = h_cached.to(full_out.dtype)

        # Cache miss images for next call.
        for image_hash, p0, p1, k_m, v_m in miss_k_v_per_range:
            coord.put(layer_idx, image_hash, k_m, v_m,
                      full_out[:, p0:p1, :].detach().clone())

        return full_out

    layer.forward = patched
    layer._fd_streaming_lm_wrapped = True


def install(vlm: nn.Module, *, max_entries_per_layer: int = 16) -> _LMStreamCoordinator:
    """Wrap every text-decoder layer for streaming pre-RoPE K/V +
    hidden_out caching. Returns the coordinator so streaming_vision can
    publish per-image layouts each call."""
    if getattr(vlm, _MARKER, False):
        return vlm._fd_streaming_lm_coord

    coord = _LMStreamCoordinator(max_entries_per_layer=max_entries_per_layer)
    layers = vlm.model.language_model.layers
    for i, layer in enumerate(layers):
        _wrap_layer(layer, layer_idx=i, coord=coord)

    vlm._fd_streaming_lm_coord = coord
    setattr(vlm, _MARKER, True)
    LOGGER.info("streaming_lm installed on %d layers", len(layers))
    return coord
