#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-stage latency breakdown of Alpamayo 1.5 to figure out which
FlashDrive optimisations have headroom on our hardware."""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch


def _build_inputs(model, device, *, camera_indices: list[int]):
    from alpamayo1_5 import helper
    n_cam = len(camera_indices)
    images = torch.randint(0, 256, (n_cam * 4, 3, 560, 1008), dtype=torch.uint8, device=device)
    msgs = helper.create_message(
        images,
        camera_indices=torch.tensor(camera_indices, dtype=torch.int64),
        num_frames_per_camera=4,
    )
    proc = helper.get_processor(model.tokenizer)
    batch = proc.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt", device="cuda",
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--camera-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-gen-tokens", type=int, default=32)
    ap.add_argument("--diffusion-steps", type=int, default=10)
    args = ap.parse_args()

    from accelerate import init_empty_weights
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5.flashdrive.load_helpers import (
        load_full_state_dict,
        patch_alpamayo15_config_for_qwen3vl,
    )

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"loading {args.teacher} ...")
    patch_alpamayo15_config_for_qwen3vl(args.teacher)
    cfg = Alpamayo1_5Config.from_pretrained(args.teacher)
    with init_empty_weights(include_buffers=False):
        model = Alpamayo1_5(cfg)
    model = model.to(torch.bfloat16)
    state = load_full_state_dict(args.teacher)
    model.load_state_dict(state, strict=False, assign=True)
    if any(p.is_meta for p in model.parameters()):
        model = model.to_empty(device=device)
    else:
        model = model.to(device)
    model.eval()
    model.diffusion.num_inference_steps = args.diffusion_steps

    batch = _build_inputs(model, device, camera_indices=args.camera_indices)
    print(f"input_ids len = {batch['input_ids'].shape[1]}, "
          f"pixel_values shape = {batch['pixel_values'].shape}")

    vlm = model.vlm

    # --- 1. Vision encode only ---
    def _vision():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return vlm.model.visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])

    # --- 2. VLM prefill only (model forward, no decode) ---
    from transformers import DynamicCache
    def _prefill():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache = DynamicCache()
            vlm.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

    # --- 3. VLM generate (32 forced tokens — apples-to-apples decode) ---
    def _gen():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            vlm.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                max_new_tokens=args.max_gen_tokens,
                min_new_tokens=args.max_gen_tokens,
                do_sample=False,
            )

    # --- 4. Full sample_trajectories ---
    def _full():
        td = dict(batch)
        data = {
            "tokenized_data": td,
            "ego_history_xyz": torch.zeros(1, 1, 16, 3, device=device),
            "ego_history_rot": torch.eye(3, device=device).repeat(1, 1, 16, 1, 1),
        }
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.sample_trajectories_from_data_with_vlm_rollout(
                data=data, top_p=1.0, temperature=1.0,
                num_traj_samples=1, num_traj_sets=1,
                max_generation_length=args.max_gen_tokens, return_extra=False,
            )

    def _time(label, fn, n=args.n):
        for _ in range(2): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        print(f"  {label:>32s}: {st.median(ts):8.1f} ms  (min={min(ts):.1f}, max={max(ts):.1f})")
        return st.median(ts)

    print("\nPer-stage breakdown (median over 3):")
    encode_ms = _time("vision encode (visual only)", _vision)
    prefill_ms = _time("VLM prefill (model.forward)", _prefill)
    gen_ms = _time(f"VLM generate ({args.max_gen_tokens} tok)", _gen)
    full_ms = _time("FULL sample_trajectories", _full)
    print(f"\n  derived decode (gen − prefill): {gen_ms - prefill_ms:.1f} ms")
    print(f"  derived expert (full − gen):    {full_ms - gen_ms:.1f} ms")


if __name__ == "__main__":
    main()
