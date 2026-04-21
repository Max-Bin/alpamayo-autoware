#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end bench for PARO W4A8 quantisation on Alpamayo 1.5.

Loads ``models/Alpamayo-1.5-10B-finetuned-PARO`` (the actual quantized
checkpoint) and compares against the BF16 baseline on real T4 rosbag
camera frames. Reports per-stage and end-to-end timings.

Usage:
    PYTHONPATH=src python scripts/bench_paro.py \
        --teacher models/Alpamayo-1.5-10B-finetuned \
        --paro    models/Alpamayo-1.5-10B-finetuned-PARO \
        --bag /media/binwang/COMLOPS/2026-03-02/<bag-dir>
"""
from __future__ import annotations

import argparse
import statistics as st
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

DEFAULT_CAMERA_TOPICS = [
    ("/sensing/camera/camera0/image_raw/compressed", 0),
    ("/sensing/camera/camera1/image_raw/compressed", 1),
    ("/sensing/camera/camera2/image_raw/compressed", 2),
    ("/sensing/camera/camera7/image_raw/compressed", 6),
]
NATIVE_HW = (560, 1008)


def _read_one_frame_per_camera(bag_dir: str, topics: list[str], n_frames: int):
    needed = {t: n_frames for t in topics}
    out: dict[str, list[np.ndarray]] = {t: [] for t in topics}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(bag_dir)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in needed]
        for conn, _ts, raw in reader.messages(connections=conns):
            if needed[conn.topic] == 0:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != NATIVE_HW:
                img = cv2.resize(img, (NATIVE_HW[1], NATIVE_HW[0]), interpolation=cv2.INTER_AREA)
            out[conn.topic].append(img)
            needed[conn.topic] -= 1
            if all(v == 0 for v in needed.values()):
                break
    return out


def _build_inputs(model, device, *, bag_dir, n_frames=4):
    from alpamayo1_5 import helper
    topic_idx = list(DEFAULT_CAMERA_TOPICS)
    topics = [t for t, _ in topic_idx]
    cam_idx = torch.tensor([i for _, i in topic_idx], dtype=torch.int64)
    frames_by_topic = _read_one_frame_per_camera(bag_dir, topics, n_frames=n_frames)
    frames = []
    for t, _ in topic_idx:
        for img in frames_by_topic[t]:
            frames.append(torch.from_numpy(img).permute(2, 0, 1).contiguous())
    image_frames = torch.stack(frames, dim=0).to(device)
    msgs = helper.create_message(image_frames, camera_indices=cam_idx, num_frames_per_camera=n_frames)
    proc = helper.get_processor(model.tokenizer)
    batch = proc.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt", device="cuda",
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def _load_model(teacher, dtype, device):
    from accelerate import init_empty_weights
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5.flashdrive.load_helpers import (
        load_full_state_dict, patch_alpamayo15_config_for_qwen3vl,
    )
    patch_alpamayo15_config_for_qwen3vl(teacher)
    cfg = Alpamayo1_5Config.from_pretrained(teacher)
    with init_empty_weights(include_buffers=False):
        m = Alpamayo1_5(cfg)
    m = m.to(dtype)
    state = load_full_state_dict(teacher)
    m.load_state_dict(state, strict=False, assign=True)
    if any(p.is_meta for p in m.parameters()):
        m = m.to_empty(device=device)
    else:
        m = m.to(device)
    return m.eval()


def _run(model, batch, ego_xyz, ego_rot, *, max_gen, num_steps):
    model.diffusion.num_inference_steps = num_steps
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data_with_vlm_rollout(
            data={"tokenized_data": batch, "ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot},
            top_p=1.0, temperature=1.0, num_traj_samples=1, num_traj_sets=1,
            max_generation_length=max_gen, return_extra=False,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--paro", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--max-gen-tokens", type=int, default=16)
    ap.add_argument("--diffusion-steps", type=int, default=5)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    ego_xyz = torch.zeros(1, 1, 16, 3, device=device)
    ego_rot = torch.eye(3, device=device).repeat(1, 1, 16, 1, 1)

    print(f"loading BF16 baseline from {args.teacher} ...")
    t = time.perf_counter()
    model = _load_model(args.teacher, torch.bfloat16, device)
    print(f"  loaded in {time.perf_counter()-t:.1f}s")
    batch = _build_inputs(model, device, bag_dir=args.bag)

    print("\n[BF16 baseline]")
    for _ in range(2):
        _run(model, batch, ego_xyz, ego_rot, max_gen=args.max_gen_tokens, num_steps=args.diffusion_steps)
    bf16_ts = []
    for i in range(args.n):
        ms = _run(model, batch, ego_xyz, ego_rot, max_gen=args.max_gen_tokens, num_steps=args.diffusion_steps)
        bf16_ts.append(ms)
        print(f"  run {i}: {ms:.1f} ms")
    bf16_p50 = st.median(bf16_ts)
    print(f"  p50 = {bf16_p50:.1f} ms")

    del model
    torch.cuda.empty_cache()

    print(f"\nloading + patching with PARO W4A8 from {args.paro} ...")
    t = time.perf_counter()
    # Build the BF16 model first (needs full state dict for non-quantized parts)
    # Load model in BF16 (the rest of Alpamayo uses BF16 + autocast). PARO
    # Linears cast in/out to fp16 internally per-call. Earlier code loaded
    # everything as fp16 which caused per-call recasts under the autocast.
    model = _load_model(args.teacher, torch.bfloat16, device)
    from alpamayo1_5.flashdrive import quant_paro
    n_swapped = quant_paro.install(model, ckpt_path=args.paro)
    print(f"  installed in {time.perf_counter()-t:.1f}s, {n_swapped} linears swapped")
    batch = _build_inputs(model, device, bag_dir=args.bag)

    print("\n[PARO W4A8]")
    for _ in range(2):
        _run(model, batch, ego_xyz, ego_rot, max_gen=args.max_gen_tokens, num_steps=args.diffusion_steps)
    paro_ts = []
    for i in range(args.n):
        ms = _run(model, batch, ego_xyz, ego_rot, max_gen=args.max_gen_tokens, num_steps=args.diffusion_steps)
        paro_ts.append(ms)
        print(f"  run {i}: {ms:.1f} ms")
    paro_p50 = st.median(paro_ts)
    print(f"  p50 = {paro_p50:.1f} ms")

    print("\n=== summary ===")
    print(f"  BF16 baseline : {bf16_p50:7.1f} ms")
    print(f"  PARO W4A8     : {paro_p50:7.1f} ms")
    print(f"  speedup       : {bf16_p50 / paro_p50:.2f}x")


if __name__ == "__main__":
    main()
