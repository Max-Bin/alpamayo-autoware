#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end FlashDrive vs baseline benchmark on Alpamayo 1.5.

Reports per-stage latency under each combination of FlashDrive
optimisations so we can attribute the speedup component-by-component.

Usage:
    PYTHONPATH=src python scripts/bench_flashdrive.py \
        --teacher models/Alpamayo-1.5-10B-finetuned \
        --camera-indices 0 1 2 3 \
        --num-clips 5
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch


def _build_inputs(model, device, *, camera_indices: list[int]):
    from alpamayo1_5 import helper

    n_cam = len(camera_indices)
    images = torch.randint(
        0, 256, (n_cam * 4, 3, 560, 1008), dtype=torch.uint8, device=device
    )
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
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    return {
        "tokenized_data": batch,
        "ego_history_xyz": torch.zeros(1, 1, 16, 3, device=device),
        "ego_history_rot": torch.eye(3, device=device).repeat(1, 1, 16, 1, 1),
    }


def _run(model, data, *, n, max_gen):
    def _kw():
        td = dict(data["tokenized_data"])
        return dict(
            data={"tokenized_data": td,
                  "ego_history_xyz": data["ego_history_xyz"],
                  "ego_history_rot": data["ego_history_rot"]},
            top_p=1.0, temperature=1.0,
            num_traj_samples=1, num_traj_sets=1,
            max_generation_length=max_gen,
            return_extra=False,
        )

    # Warmup
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2):
            model.sample_trajectories_from_data_with_vlm_rollout(**_kw())
    torch.cuda.synchronize()

    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(**_kw())
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def _load(teacher, dtype, device):
    """Load Alpamayo 1.5 without needing access to gated nvidia/Cosmos-Reason2-8B.

    Cosmos-Reason2-8B is post-trained from Qwen/Qwen3-VL-8B-Instruct with
    the SAME architecture (per their HF README). The Alpamayo-1.5
    checkpoint contains the actual VLM weights — we only need the
    architecture spec, which Qwen3-VL-8B-Instruct provides publicly.
    The patch_embed reshape (1152x1536 -> 1152x3x2x16x16) is element-
    identical and stored differently between Cosmos and Qwen3-VL.
    """
    from accelerate import init_empty_weights
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5.flashdrive.load_helpers import (
        load_full_state_dict,
        patch_alpamayo15_config_for_qwen3vl,
    )
    patch_alpamayo15_config_for_qwen3vl(teacher)
    cfg = Alpamayo1_5Config.from_pretrained(teacher)
    with init_empty_weights(include_buffers=False):
        model = Alpamayo1_5(cfg)
    model = model.to(dtype)
    state = load_full_state_dict(teacher)
    model.load_state_dict(state, strict=False, assign=True)
    if any(p.is_meta for p in model.parameters()):
        model = model.to_empty(device=device)
    else:
        model = model.to(device)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--camera-indices", type=int, nargs="+", default=[1])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-gen-tokens", type=int, default=32)
    ap.add_argument("--diffusion-steps", type=int, default=5)
    ap.add_argument(
        "--configs", nargs="+",
        default=["baseline",
                 "adaptive_flow",
                 "kernel_fusion_qkv kernel_fusion_mlp",
                 "adaptive_flow kernel_fusion_qkv kernel_fusion_mlp"],
        help='space-separated FlashDrive options per group; "baseline" disables all',
    )
    args = ap.parse_args()

    from alpamayo1_5.flashdrive import FlashDriveConfig, apply_flashdrive

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"loading {args.teacher} ...")
    t0 = time.perf_counter()
    model = _load(args.teacher, torch.bfloat16, device)
    model.diffusion.num_inference_steps = args.diffusion_steps
    print(f"  loaded in {time.perf_counter()-t0:.1f}s, diffusion_steps={args.diffusion_steps}")

    data = _build_inputs(model, device, camera_indices=args.camera_indices)

    results: dict[str, list[float]] = {}
    baseline_p50 = None
    for raw in args.configs:
        opts = raw.split()
        label = "baseline" if opts == ["baseline"] else "+".join(opts)
        print(f"\n--- config: {label} ---")
        # Reload model fresh for each config so previous installs don't leak.
        del model; torch.cuda.empty_cache()
        model = _load(args.teacher, torch.bfloat16, device)
        model.diffusion.num_inference_steps = args.diffusion_steps
        if opts != ["baseline"]:
            cfg = FlashDriveConfig.from_strings(opts)
            apply_flashdrive(model, cfg)
        ts = _run(model, data, n=args.n, max_gen=args.max_gen_tokens)
        p50 = st.median(ts)
        if baseline_p50 is None:
            baseline_p50 = p50
        speedup = baseline_p50 / p50
        print(f"  p50={p50:7.1f} ms  min={min(ts):7.1f}  max={max(ts):7.1f}  speedup={speedup:.2f}x")
        results[label] = ts

    print("\n=== summary ===")
    for label, ts in results.items():
        p50 = st.median(ts)
        speedup = baseline_p50 / p50
        print(f"  {label:>60s}: {p50:7.1f} ms  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
