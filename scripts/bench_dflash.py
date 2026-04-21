#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DFlash Phase-2 bench — measure mean_accept on real rosbag input.

Loads Alpamayo 1.5 target + DFlash draft, runs block-diffusion
speculative decoding, reports:
    - mean_accept      — paper target is ~4–5 (max 8 with block_size=8)
    - tokens generated — for correctness sanity
    - e2e latency comparison vs HF generate() greedy

If mean_accept > ~2, it's worth integrating with FlashDrive. Below
1.5 means the draft is usually producing wrong tokens; re-verify
the load / forward.

Usage:
    PYTHONPATH=src .venv-bench/bin/python3 scripts/bench_dflash.py \\
        --teacher models/Alpamayo-1.5-10B-finetuned \\
        --draft   models/Alpamayo-1.5-DFlash \\
        --bag     /media/binwang/COMLOPS/2026-03-02/<bag> \\
        --max-gen-tokens 16
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--max-gen-tokens", type=int, default=16)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy; 0.6 matches draft's training distribution")
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from bench_paro import _build_inputs, _load_model
    from alpamayo1_5.flashdrive.dflash_spec import dflash_generate_vlm, load_dflash_draft

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"loading target {args.teacher} ...")
    t0 = time.perf_counter()
    model = _load_model(args.teacher, torch.bfloat16, device)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\nloading DFlash draft {args.draft} ...")
    t0 = time.perf_counter()
    draft = load_dflash_draft(args.draft, device=device, dtype=torch.bfloat16)
    print(f"  draft loaded in {time.perf_counter()-t0:.1f}s  "
          f"(block_size={draft.block_size}, target_layer_ids={list(draft.target_layer_ids)})")

    batch = _build_inputs(model, device, bag_dir=args.bag)
    input_ids = batch.pop("input_ids")
    P = input_ids.shape[1]
    print(f"prefix length P = {P}")

    vlm = model.vlm

    # --- HF greedy baseline for comparison
    print(f"\n[baseline] vlm.generate greedy × {args.max_gen_tokens} tokens ...")
    baselines = []
    for _ in range(2):  # warmup
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            vlm.model.rope_deltas = None
            _ = vlm.generate(input_ids=input_ids, max_new_tokens=args.max_gen_tokens,
                             do_sample=False, use_cache=True, **batch)
    for _ in range(args.n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            vlm.model.rope_deltas = None
            base_out = vlm.generate(input_ids=input_ids, max_new_tokens=args.max_gen_tokens,
                                    do_sample=False, use_cache=True, **batch)
        torch.cuda.synchronize()
        baselines.append((time.perf_counter() - t0) * 1000)
    base_p50 = st.median(baselines)
    print(f"  p50 = {base_p50:.1f} ms  ({args.max_gen_tokens} tokens)")
    base_tokens = base_out[0, P:].tolist()
    print(f"  baseline tokens: {base_tokens[:16]}")

    # --- DFlash spec-decode
    print(f"\n[dflash] block-diffusion spec-decode × {args.max_gen_tokens} new tokens ...")

    # First run with debug on to see per-block draft predictions.
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO, format="%(message)s")
    print("  --- one debug run (verbose) ---")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _ = dflash_generate_vlm(
            draft=draft, vlm=vlm, input_ids=input_ids, image_kwargs=batch,
            max_new_tokens=args.max_gen_tokens, temperature=0.0, return_stats=True,
            debug=True,
        )
    print("  --- end debug run ---\n")
    _lg.getLogger().setLevel(_lg.WARNING)

    # warmup
    for _ in range(2):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = dflash_generate_vlm(
                draft=draft, vlm=vlm, input_ids=input_ids, image_kwargs=batch,
                max_new_tokens=args.max_gen_tokens, temperature=args.temperature, return_stats=True,
            )
    speeds = []
    last_out = None
    for i in range(args.n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            last_out = dflash_generate_vlm(
                draft=draft, vlm=vlm, input_ids=input_ids, image_kwargs=batch,
                max_new_tokens=args.max_gen_tokens, temperature=args.temperature, return_stats=True,
            )
        torch.cuda.synchronize()
        speeds.append((time.perf_counter() - t0) * 1000)
        print(f"  run {i}: {speeds[-1]:.1f} ms   "
              f"accept_lens={last_out.acceptance_lengths}   "
              f"mean={last_out.mean_accept:.2f}")

    dflash_p50 = st.median(speeds)
    dflash_tokens = last_out.output_ids[0, P:P + args.max_gen_tokens].tolist()
    agree = sum(1 for a, b in zip(base_tokens[:len(dflash_tokens)], dflash_tokens) if a == b)

    print("\n=== summary ===")
    print(f"  baseline HF greedy    : {base_p50:7.1f} ms")
    print(f"  DFlash spec-decode    : {dflash_p50:7.1f} ms  speedup={base_p50/dflash_p50:.2f}x")
    print(f"  mean_accept           : {last_out.mean_accept:.2f} "
          f"(paper target ~4-5; > 2 means draft is helping)")
    print(f"  greedy-agreement      : {agree}/{len(dflash_tokens)} tokens")
    if last_out.mean_accept < 1.2:
        print("\n!! mean_accept is still near 1.0 — draft not contributing.")
        print("   Check the draft load / forward: maybe mask_embedding not injected,")
        print("   or noise_embedding being built with wrong embed_tokens.")


if __name__ == "__main__":
    main()
