#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bench: streaming-window ViT cache.

Simulates a real ROS inference cadence where 3 of 4 frames overlap with
the previous step. Measures vision-encode latency with and without
``flashdrive.streaming_vision``.
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--num-cameras", type=int, default=4)
    ap.add_argument("--frames-per-camera", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=8,
                    help="number of streaming inference steps to simulate")
    args = ap.parse_args()

    from accelerate import init_empty_weights
    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5.flashdrive.load_helpers import (
        load_full_state_dict, patch_alpamayo15_config_for_qwen3vl,
    )
    from alpamayo1_5.flashdrive.streaming_vision import install as install_streaming

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"loading {args.teacher} ...")
    patch_alpamayo15_config_for_qwen3vl(args.teacher)
    cfg = Alpamayo1_5Config.from_pretrained(args.teacher)
    with init_empty_weights(include_buffers=False):
        m = Alpamayo1_5(cfg)
    m = m.to(torch.bfloat16)
    state = load_full_state_dict(args.teacher)
    m.load_state_dict(state, strict=False, assign=True)
    if any(p.is_meta for p in m.parameters()):
        m = m.to_empty(device=device)
    else:
        m = m.to(device)
    m.eval()

    # Generate a sliding pool of "fresh" frames, each unique. At each step
    # we take the last `frames_per_camera` from a sliding window of length
    # `n_steps + frames_per_camera - 1`.
    pool = torch.randint(
        0, 256,
        (args.num_cameras, args.n_steps + args.frames_per_camera - 1, 3, 560, 1008),
        dtype=torch.uint8, device=device,
    )

    proc = helper.get_processor(m.tokenizer)

    def _encode_step(window_offset: int) -> float:
        # pull frames [window_offset : window_offset + frames_per_camera]
        # for each camera, flatten to [num_cam * fpc, 3, H, W]
        frames = pool[:, window_offset:window_offset + args.frames_per_camera]
        flat = frames.reshape(args.num_cameras * args.frames_per_camera, 3, 560, 1008)
        msgs = helper.create_message(
            flat,
            camera_indices=torch.tensor(list(range(args.num_cameras)), dtype=torch.int64),
            num_frames_per_camera=args.frames_per_camera,
        )
        batch = proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt", device="cuda",
        )
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            m.vlm.model.visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    # Baseline (no cache)
    print("\n[baseline] vision encode per streaming step (no cache):")
    base_ts = []
    for s in range(args.n_steps):
        ms = _encode_step(s)
        base_ts.append(ms)
    print(f"  per-step ms = {[f'{t:.1f}' for t in base_ts]}")
    print(f"  median = {st.median(base_ts):.1f} ms")

    # With cache
    print("\n[streaming_vision] vision encode per streaming step:")
    cache = install_streaming(m.vlm, window=args.num_cameras * args.frames_per_camera + 4)
    cache_ts = []
    for s in range(args.n_steps):
        ms = _encode_step(s)
        cache_ts.append(ms)
    print(f"  per-step ms = {[f'{t:.1f}' for t in cache_ts]}")
    print(f"  median = {st.median(cache_ts):.1f} ms")
    print(f"  cache hits/misses = {cache.hits}/{cache.misses}")

    speedup = st.median(base_ts) / st.median(cache_ts)
    print(f"\nstreaming_vision encode speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
