#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end streaming bench on multi-camera rosbag frames.

Simulates the ROS callback cadence: at 10 Hz, 3 of every 4 frames overlap
between consecutive inference steps. Reports per-step e2e wall-clock
under several FlashDrive configs, including streaming_vision which only
shows its real value when consecutive calls share frames.

Usage:
    PYTHONPATH=src python scripts/bench_streaming_e2e.py \
        --teacher models/Alpamayo-1.5-10B-finetuned \
        --bag /path/to/rosbag \
        --steps 8
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

# Camera→model_index: matches load_physical_aiavdataset's convention.
DEFAULT_CAMERA_TOPICS = [
    ("/sensing/camera/camera0/image_raw/compressed", 0),  # cross_left
    ("/sensing/camera/camera1/image_raw/compressed", 1),  # front_wide
    ("/sensing/camera/camera2/image_raw/compressed", 2),  # cross_right
    ("/sensing/camera/camera7/image_raw/compressed", 6),  # front_tele
]
NATIVE_HW = (560, 1008)


def _read_streaming_pool(bag_dir: str, topics: list[str], n_total: int):
    """Read ``n_total`` frames per topic from the bag at the head."""
    out: dict[str, list[np.ndarray]] = {t: [] for t in topics}
    needed = {t: n_total for t in topics}
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


def _build_step_inputs(model, device, *, frames_by_topic, step_idx: int, frames_per_camera: int):
    """For step S, take frames [S, S+1, S+2, S+3] per camera (sliding window)."""
    from alpamayo1_5 import helper

    topic_idx = list(DEFAULT_CAMERA_TOPICS)
    topics = [t for t, _ in topic_idx]
    cam_idx = torch.tensor([i for _, i in topic_idx], dtype=torch.int64)

    frames = []
    for t in topics:
        for off in range(frames_per_camera):
            img = frames_by_topic[t][step_idx + off]
            frames.append(torch.from_numpy(img).permute(2, 0, 1).contiguous())
    image_frames = torch.stack(frames, dim=0).to(device)

    msgs = helper.create_message(
        image_frames,
        camera_indices=cam_idx,
        num_frames_per_camera=frames_per_camera,
    )
    proc = helper.get_processor(model.tokenizer)
    batch = proc.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt", device="cuda",
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def _load(teacher, dtype, device):
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


def _run_step(model, batch, ego_xyz, ego_rot, *, max_gen, num_steps):
    model.diffusion.num_inference_steps = num_steps
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data={"tokenized_data": batch,
                  "ego_history_xyz": ego_xyz,
                  "ego_history_rot": ego_rot},
            top_p=1.0, temperature=1.0,
            num_traj_samples=1, num_traj_sets=1,
            max_generation_length=max_gen,
            return_extra=False,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--steps", type=int, default=8,
                    help="number of streaming inference steps to simulate")
    ap.add_argument("--frames-per-camera", type=int, default=4)
    ap.add_argument("--max-gen-tokens", type=int, default=16)
    ap.add_argument("--diffusion-steps", type=int, default=5)
    ap.add_argument("--configs", nargs="+",
                    default=["baseline",
                             "streaming_vision",
                             "adaptive_flow kernel_fusion_qkv kernel_fusion_mlp",
                             "adaptive_flow kernel_fusion_qkv kernel_fusion_mlp streaming_vision"])
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    from alpamayo1_5.flashdrive import FlashDriveConfig, apply_flashdrive

    print(f"reading {args.steps + args.frames_per_camera - 1} frames per camera from bag ...")
    topics = [t for t, _ in DEFAULT_CAMERA_TOPICS]
    frames_by_topic = _read_streaming_pool(
        args.bag, topics, args.steps + args.frames_per_camera - 1,
    )
    for t, lst in frames_by_topic.items():
        print(f"  {t}: {len(lst)} frames")

    ego_xyz = torch.zeros(1, 1, 16, 3, device=device)
    ego_rot = torch.eye(3, device=device).repeat(1, 1, 16, 1, 1)

    print(f"\nloading {args.teacher} ...")
    t0 = time.perf_counter()
    model = _load(args.teacher, torch.bfloat16, device)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    results = {}
    baseline_p50 = None
    for raw in args.configs:
        opts = raw.split()
        label = "baseline" if opts == ["baseline"] else "+".join(opts)
        print(f"\n=== config: {label} ===")
        # Reload fresh to avoid cumulative install effects
        del model; torch.cuda.empty_cache()
        model = _load(args.teacher, torch.bfloat16, device)
        if opts != ["baseline"]:
            cfg = FlashDriveConfig.from_strings(opts)
            apply_flashdrive(model, cfg)

        # Warmup with step 0
        batch0 = _build_step_inputs(
            model, device, frames_by_topic=frames_by_topic, step_idx=0,
            frames_per_camera=args.frames_per_camera,
        )
        for _ in range(2):
            _run_step(model, batch0, ego_xyz, ego_rot,
                      max_gen=args.max_gen_tokens, num_steps=args.diffusion_steps)

        ts = []
        for s in range(args.steps):
            batch = _build_step_inputs(
                model, device, frames_by_topic=frames_by_topic, step_idx=s,
                frames_per_camera=args.frames_per_camera,
            )
            ms = _run_step(model, batch, ego_xyz, ego_rot,
                           max_gen=args.max_gen_tokens, num_steps=args.diffusion_steps)
            ts.append(ms)
        p50 = st.median(ts)
        if baseline_p50 is None:
            baseline_p50 = p50
        speedup = baseline_p50 / p50
        print(f"  per-step ms: {[f'{t:.0f}' for t in ts]}")
        print(f"  p50={p50:.1f}  min={min(ts):.1f}  max={max(ts):.1f}  speedup={speedup:.2f}x")
        if hasattr(model.vlm, "_fd_vit_cache"):
            cache = model.vlm._fd_vit_cache
            print(f"  vit cache hits/misses = {cache.hits}/{cache.misses}")
        results[label] = ts

    print("\n=== summary ===")
    for label, ts in results.items():
        p50 = st.median(ts)
        speedup = baseline_p50 / p50
        print(f"  {label:>70s}: p50={p50:7.1f} ms  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
