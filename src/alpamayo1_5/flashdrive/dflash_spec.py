# SPDX-License-Identifier: Apache-2.0
"""DFlash block-diffusion speculative decoding — VLM-aware adapter.

Upstream ``dflash.model.dflash_generate`` assumes the target is a plain
``Qwen3ForCausalLM`` (linear position_ids, no images). Our target is
``Qwen3VLForConditionalGeneration`` (3-axis MRoPE position_ids derived
from image_grid_thw, rope_deltas cached on the model). This module
adapts the upstream control flow to that target while keeping the draft
forward (which is a plain 2-layer Qwen3) unchanged.

Flow:
    1. Prefill: vlm(input_ids, pixel_values, image_grid_thw, ...) with
       past_key_values=None. HF's Qwen3VL get_rope_index computes
       position_ids (3-axis) and stores rope_deltas on the model.
    2. Verify step: vlm(input_ids=block_tokens, past_kv=cache,
       cache_position=[start..start+block_size]). HF's non-prefill
       branch reconstructs position_ids from cache_position +
       rope_deltas (all 3 MRoPE axes identical post-image, which is
       what we want for generated text tokens).
    3. Draft step: draft(target_hidden, noise_embedding, position_ids=
       linear_slice, past_kv=draft_cache). Draft is plain Qwen3, no MRoPE.

The acceptance logic itself is copied from upstream (cumulative prefix
match between drafted and target-greedy-argmax tokens), which is the
greedy variant of standard speculative decoding.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from dflash.model import DFlashDraftModel, extract_context_feature, sample

LOGGER = logging.getLogger(__name__)


def load_dflash_draft(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> DFlashDraftModel:
    """Load a DFlash draft from ``models/Alpamayo-1.5-DFlash``.

    Assembles a ``Qwen3Config`` with the extra ``dflash_config`` block so
    :class:`DFlashDraftModel` can pick up ``target_layer_ids`` and
    ``mask_token_id`` from it. Loads the 25 tensors + the (4096,) shared
    mask embedding (``mask_embedding.pt``).
    """
    ckpt = Path(ckpt_path)
    with open(ckpt / "config.json") as f:
        raw = json.load(f)

    dflash_config = {
        "target_layer_ids": raw["target_layer_ids"],
        "mask_token_id": raw["mask_token_id"],
    }
    base_cfg = {k: v for k, v in raw.items()
                if k not in ("architectures", "target_layer_ids", "num_target_layers",
                             "mask_token_id", "block_size", "context_len",
                             "resolution_scale", "dtype")}
    cfg = Qwen3Config(**base_cfg)
    cfg.num_target_layers = raw["num_target_layers"]
    cfg.block_size = raw["block_size"]
    cfg.dflash_config = dflash_config
    cfg._attn_implementation = "sdpa"

    draft = DFlashDraftModel(cfg).to(dtype).to(device)

    from safetensors.torch import load_file
    state = load_file(str(ckpt / "model.safetensors"))
    mask_emb = torch.load(ckpt / "mask_embedding.pt", map_location="cpu", weights_only=True)
    missing, unexpected = draft.load_state_dict(state, strict=False)
    if unexpected:
        LOGGER.warning("DFlash draft: unexpected keys %s", unexpected)
    # mask_embedding is a buffer not in the state_dict layout; stash it on the draft.
    draft.mask_embedding = mask_emb.to(dtype=dtype, device=device)
    LOGGER.info("DFlash draft: loaded %d tensors, %d missing (%s...)",
                len(state), len(missing), missing[:3] if missing else "")
    return draft.eval()


@torch.inference_mode()
def dflash_generate_vlm(
    draft: DFlashDraftModel,
    vlm: nn.Module,
    input_ids: torch.LongTensor,
    *,
    image_kwargs: dict,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]] = None,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    return_stats: bool = True,
    debug: bool = False,
) -> SimpleNamespace:
    """VLM-aware block-diffusion speculative decode.

    Args:
        draft: loaded DFlashDraftModel (2 layers, fed target hiddens at
            layers ``[24, 30, 31, 32, 34]``).
        vlm: Qwen3VLForConditionalGeneration (the target).
        input_ids: (1, P) prefix tokens (must include image tokens in
            the right slots — produced by apply_chat_template).
        image_kwargs: all non-input_ids outputs of apply_chat_template
            (pixel_values, image_grid_thw, attention_mask, etc.).
        max_new_tokens: total tokens to generate (including the eagerly
            sampled first token).
        stop_token_ids: early-stop tokens (optional).
        temperature: 0 for greedy (what DFlash paper reports).

    Returns:
        SimpleNamespace(output_ids, acceptance_lengths, ...). The
        ``mean_accept = mean(acceptance_lengths)`` is the key metric
        for debugging — paper reports ~4–5 on plain LM, so anything
        above 1.0 means draft is actually helping.
    """
    P = input_ids.shape[1]
    max_length = P + max_new_tokens
    block_size = draft.block_size if block_size is None else block_size
    mask_token_id = draft.mask_token_id if mask_token_id is None else mask_token_id

    device = vlm.device
    # Preallocate output buffer with mask_token_id — the block slice [start:start+block_size]
    # always has accepted[0] + mask[1:block_size]; the draft rewrites [1:] on each iter.
    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device,
    )
    # Draft-side position_ids. Built lazily after rope_delta is known below.
    draft_position_ids = None

    past_kv_target = DynamicCache()
    past_kv_draft = DynamicCache()

    # --- Target prefill. Reset rope_deltas so get_rope_index recomputes; pass
    # pixel_values + image_grid_thw exactly like bench_paro does in Test A.
    vlm.model.rope_deltas = None
    out = vlm(
        input_ids=input_ids,
        past_key_values=past_kv_target,
        use_cache=True,
        output_hidden_states=(block_size > 1),
        **image_kwargs,
    )
    # Capture the MRoPE delta HF computed during prefill. For text tokens past
    # the last image, all 3 MRoPE axes collapse to (linear_pos + rope_delta),
    # which puts generated-text positions back in the draft's training range
    # (draft is plain Qwen3, trained with low linear positions).
    rope_delta = int(vlm.model.rope_deltas[0].item())
    if debug:
        LOGGER.info("rope_delta = %d  (draft will see positions shifted by this)", rope_delta)
    draft_position_ids = (torch.arange(output_ids.shape[1], device=device) + rope_delta).unsqueeze(0)
    output_ids[:, :P] = input_ids
    output_ids[:, P:P + 1] = sample(out.logits[:, -1:, :], temperature)
    if block_size > 1:
        if debug:
            nl = len(out.hidden_states)
            LOGGER.info(
                "hidden_states: %d tensors, each shape %s; picking layers %s (+1 offset)",
                nl, tuple(out.hidden_states[0].shape), list(draft.target_layer_ids),
            )
            for lid in draft.target_layer_ids:
                h = out.hidden_states[lid + 1][0, -1]
                LOGGER.info("  layer %d last-pos: mean=%.4f std=%.4f min=%.4f max=%.4f "
                            "|| layer %d (+0 offset) last-pos std=%.4f",
                            lid, h.float().mean().item(), h.float().std().item(),
                            h.float().min().item(), h.float().max().item(),
                            lid, out.hidden_states[lid][0, -1].float().std().item())
        target_hidden = extract_context_feature(out.hidden_states, draft.target_layer_ids)
        if debug:
            LOGGER.info("target_hidden shape=%s  norm=%.3f  (concat of 5×%d)",
                        tuple(target_hidden.shape),
                        target_hidden[0, -1].float().norm().item(), target_hidden.shape[-1] // 5)
    else:
        target_hidden = None

    acceptance_lengths: list[int] = []
    start = P
    while start < max_length:
        block_slot = output_ids[:, start : start + block_size].clone()  # first is accepted, rest mask
        if block_size > 1:
            # Mask token id is outside target's vocab (155697 == vocab size); DFlash trained
            # a separate shared mask embedding for it (mask_embedding.pt, (hidden,)).
            # Embed real tokens via target's table, then overwrite mask positions with mask_emb.
            is_mask = (block_slot == mask_token_id)
            safe_slot = block_slot.masked_fill(is_mask, 0)
            noise_embedding = vlm.model.language_model.embed_tokens(safe_slot)
            noise_embedding = torch.where(
                is_mask.unsqueeze(-1), draft.mask_embedding.to(noise_embedding.dtype), noise_embedding,
            )
            draft_pos_ids = draft_position_ids[:, past_kv_draft.get_seq_length(): start + block_size]
            draft_hidden = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_pos_ids,
                past_key_values=past_kv_draft,
                use_cache=True,
                is_causal=False,
            )
            # Last block_size-1 positions correspond to the masked slots (positions 1..block_size-1).
            draft_logits = vlm.lm_head(draft_hidden[:, 1 - block_size:, :])
            past_kv_draft.crop(start)
            draft_pred = sample(draft_logits, temperature)
            if debug:
                LOGGER.info(
                    "draft at start=%d: slot_in=%s  draft_pred=%s  "
                    "top5_pos0=%s",
                    start, block_slot[0].tolist(), draft_pred[0].tolist(),
                    draft_logits[0, 0].topk(5).indices.tolist(),
                )
            block_slot[:, 1:] = draft_pred

        # --- Target verify on the block. No explicit position_ids — HF's non-prefill
        # branch will build 3-axis MRoPE from cache_position + rope_deltas.
        cache_position = torch.arange(start, start + block_size, device=device)
        out = vlm(
            input_ids=block_slot,
            past_key_values=past_kv_target,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=(block_size > 1),
        )
        # Shift-by-1 alignment (Phase 1 result): out.logits[:, j] predicts token at
        # position start+j+1. We compare draft's block_slot[:, j+1] (1..block_size-1)
        # to target-greedy's posterior[:, j] (0..block_size-2).
        posterior = sample(out.logits, temperature)
        if debug:
            LOGGER.info("    posterior=%s  compare draft[1:]=%s vs target[:-1]=%s",
                        posterior[0].tolist(), block_slot[0, 1:].tolist(),
                        posterior[0, :-1].tolist())
        # acceptance_length = prefix length of (block_slot[1:] == posterior[:-1])
        acceptance_length = (block_slot[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        # Commit accepted tokens; the posterior at accept_length gives the "resampled" next token
        output_ids[:, start : start + acceptance_length + 1] = block_slot[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_kv_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if block_size > 1:
            # Re-extract target hidden for next iter — only the accepted prefix + the
            # newly-sampled "bonus" position (accept_length+1 total).
            target_hidden = extract_context_feature(
                out.hidden_states, draft.target_layer_ids
            )[:, :acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            sid in output_ids[:, P:start] for sid in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start, max_length)]
    return SimpleNamespace(
        output_ids=output_ids,
        acceptance_lengths=acceptance_lengths,
        mean_accept=float(sum(acceptance_lengths)) / max(len(acceptance_lengths), 1),
        num_input_tokens=P,
        num_output_tokens=output_ids.shape[1] - P,
    )
