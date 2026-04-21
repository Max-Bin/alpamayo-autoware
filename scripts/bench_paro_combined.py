#!/usr/bin/env python3
"""Test PARO + the rest of the FlashDrive stack on real T4 streaming."""
from __future__ import annotations
import argparse, time, statistics as st, sys
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--paro", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-gen-tokens", type=int, default=16)
    ap.add_argument("--diffusion-steps", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    sys.path.insert(0, str(Path(__file__).parent))
    from bench_streaming_e2e import _load, _read_streaming_pool, _build_step_inputs, DEFAULT_CAMERA_TOPICS
    from alpamayo1_5.flashdrive import FlashDriveConfig, apply_flashdrive
    from alpamayo1_5.flashdrive import quant_paro

    ego_xyz = torch.zeros(1, 1, 16, 3, device=device)
    ego_rot = torch.eye(3, device=device).repeat(1, 1, 16, 1, 1)
    topics = [t for t, _ in DEFAULT_CAMERA_TOPICS]
    print("reading rosbag ...")
    frames = _read_streaming_pool(args.bag, topics, args.steps + 4)

    def run_config(label, install_fn):
        del_buffers = ["model"] if "model" in dir() else []
        torch.cuda.empty_cache()
        m = _load(args.teacher, torch.bfloat16, device)
        m.diffusion.num_inference_steps = args.diffusion_steps
        install_fn(m)
        # Warmup
        for s in range(2):
            batch = _build_step_inputs(m, device, frames_by_topic=frames, step_idx=s, frames_per_camera=4)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                m.sample_trajectories_from_data_with_vlm_rollout(
                    data={"tokenized_data": batch, "ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot},
                    top_p=1.0, temperature=1.0, num_traj_samples=1, num_traj_sets=1,
                    max_generation_length=args.max_gen_tokens, return_extra=False,
                )
        torch.cuda.synchronize()
        ts = []
        for s in range(args.steps):
            batch = _build_step_inputs(m, device, frames_by_topic=frames,
                                       step_idx=s % args.steps, frames_per_camera=4)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                m.sample_trajectories_from_data_with_vlm_rollout(
                    data={"tokenized_data": batch, "ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot},
                    top_p=1.0, temperature=1.0, num_traj_samples=1, num_traj_sets=1,
                    max_generation_length=args.max_gen_tokens, return_extra=False,
                )
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        p50 = st.median(ts); minv = min(ts); maxv = max(ts)
        print(f"{label:55s}: p50={p50:6.1f}  min={minv:6.1f}  max={maxv:6.1f}")
        return p50, minv

    print("\n=== running configs ===")
    base, _ = run_config("baseline (BF16)", lambda m: None)

    paro_only_p50, _ = run_config("PARO only", lambda m: quant_paro.install(m, ckpt_path=args.paro))

    flash_p50, flash_min = run_config(
        "adaptive+kf+streaming(v+lm)",
        lambda m: apply_flashdrive(m, FlashDriveConfig.from_strings(
            "adaptive_flow kernel_fusion_qkv kernel_fusion_mlp streaming_vision streaming_lm".split())),
    )

    flash_paro_p50, flash_paro_min = run_config(
        "PARO + adaptive+kf+streaming(v+lm)",
        lambda m: (quant_paro.install(m, ckpt_path=args.paro),
                   apply_flashdrive(m, FlashDriveConfig.from_strings(
                       "adaptive_flow kernel_fusion_qkv kernel_fusion_mlp streaming_vision streaming_lm".split()))),
    )

    print("\n=== summary ===")
    for lbl, p in [("baseline", base), ("PARO", paro_only_p50),
                   ("flashdrive (no PARO)", flash_p50), ("flashdrive + PARO", flash_paro_p50)]:
        print(f"  {lbl:35s}: p50={p:6.1f}  speedup={base/p:.2f}x")
    print(f"  best step (flashdrive)             : {flash_min:.1f} ms ({base/flash_min:.2f}x)")
    print(f"  best step (flashdrive + PARO)      : {flash_paro_min:.1f} ms ({base/flash_paro_min:.2f}x)")


if __name__ == "__main__":
    main()
