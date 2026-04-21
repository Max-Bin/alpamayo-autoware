#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 DFlash root-cause debug — NOT a bench, NOT a speculative impl.

Question under test:
    Does Alpamayo 1.5's VLM produce the SAME logits when we feed N generated
    tokens as ONE batched forward (verify-style) vs N individual forward
    steps (HF incremental, what generate() does)?

If the answers agree → the previous 0% accept rate was NOT from model
non-determinism; it was a bug in how our old speculative.py built
position_ids / cache_position / attention_mask for the verify pass.
If the answers disagree → bf16 drift is real and we need fp32 attention
accumulation (or a different verify kernel).

Three tests on the same (prefix, first-N-greedy-tokens):
    (A) Ground truth — vlm.generate(max_new_tokens=N, do_sample=False,
        output_logits=True). Incremental decode, one token per forward.
    (B) Single-prefill  — vlm(prefix + N greedy tokens, no past_kv).
        One big forward over the whole thing. Tests "batched vs
        incremental on a fresh prefill".
    (C) True verify     — run generate for 0 steps to get the prompt
        cache, then vlm(input_ids=N_tokens, past_kv=cache,
        cache_position=[P..P+N-1]). This is what a speculative verify
        call actually does.

Report for each position 0..N-1: max |Δlogit|, argmax match?, top-5
overlap, KL(inc || other).

Usage:
    PYTHONPATH=src python scripts/debug_dflash_step.py \\
        --teacher models/Alpamayo-1.5-10B-finetuned \\
        --bag /media/binwang/COMLOPS/2026-03-02/<bag-dir> \\
        --n-verify 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_helpers():
    sys.path.insert(0, str(Path(__file__).parent))
    from bench_paro import _build_inputs, _load_model
    return _build_inputs, _load_model


def _top5(logits: torch.Tensor) -> torch.Tensor:
    return logits.topk(5).indices


def _compare(name: str, ref: torch.Tensor, cand: torch.Tensor) -> None:
    """ref and cand are both (N, V)."""
    n, _ = ref.shape
    diff = (cand.float() - ref.float()).abs()
    max_abs = diff.max(dim=-1).values
    argmax_ref = ref.argmax(dim=-1)
    argmax_cand = cand.argmax(dim=-1)
    match = (argmax_ref == argmax_cand)
    top5_ref = _top5(ref)
    top5_cand = _top5(cand)
    top5_overlap = torch.stack([
        (top5_ref[i].unsqueeze(1) == top5_cand[i].unsqueeze(0)).any(dim=1).sum()
        for i in range(n)
    ]).float() / 5.0
    kl = F.kl_div(
        F.log_softmax(cand.float(), dim=-1),
        F.softmax(ref.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)

    print(f"\n=== {name} ===")
    print(f"  position |  argmax_ref  argmax_cand  match | max|Δ| | top5-overlap | KL(ref||cand)")
    print(f"  ---------+-------------------------------+--------+--------------+----------------")
    for i in range(n):
        mark = "✓" if match[i].item() else "✗"
        print(f"  {i:>8d} | {argmax_ref[i].item():>11d}  {argmax_cand[i].item():>11d}    {mark}  | "
              f"{max_abs[i].item():6.3f} |    {top5_overlap[i].item():.2f}    | {kl[i].item():9.4f}")
    print(f"  overall argmax-match-rate: {match.float().mean().item():.3f}  "
          f"({match.sum().item()}/{n})")
    print(f"  mean max|Δ|: {max_abs.mean().item():.4f}   "
          f"mean KL: {kl.mean().item():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--n-verify", type=int, default=8,
                    help="how many greedy tokens to verify (N)")
    args = ap.parse_args()

    build_inputs, load_model = _load_helpers()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"loading {args.teacher} ...")
    model = load_model(args.teacher, torch.bfloat16, device)
    print("  loaded")

    batch = build_inputs(model, device, bag_dir=args.bag)
    input_ids = batch.pop("input_ids")
    P = input_ids.shape[1]
    print(f"prefix length P = {P}")

    vlm = model.vlm
    N = args.n_verify

    # --------------------------------------------------------------------- (A)
    # Ground truth: HF incremental greedy. Capture per-step logits.
    print(f"\n[A] incremental greedy × {N} steps (ground truth) ...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Reset rope_deltas so the next forward rebuilds position_ids fresh.
        vlm.model.rope_deltas = None
        gen = vlm.generate(
            input_ids=input_ids,
            max_new_tokens=N,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True,
            **batch,
        )
    seq = gen.sequences            # (1, P+N)
    logits_A = torch.stack(gen.logits, dim=1)[0]  # (N, V)
    gen_tokens = seq[:, P:P + N]   # (1, N)
    print(f"  generated tokens: {gen_tokens[0].tolist()}")

    # --------------------------------------------------------------------- (A')
    # Sanity check: single forward on JUST the prefix. logits[P-1] should equal
    # logits_A[0] exactly (both compute "distribution for next token after prefix").
    print(f"\n[A'] sanity: single forward on prefix alone (length P={P}) ...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm.model.rope_deltas = None
        out_Ap = vlm(input_ids=input_ids, use_cache=False, **batch)
    logits_Ap = out_Ap.logits[0, P - 1:P]   # (1, V)
    print(f"  logits_A[0]  argmax={logits_A[0].argmax().item()}  "
          f"max5={_top5(logits_A[0]).tolist()}  max_val={logits_A[0].max().item():.3f}")
    print(f"  logits_A'[0] argmax={logits_Ap[0].argmax().item()}  "
          f"max5={_top5(logits_Ap[0]).tolist()}  max_val={logits_Ap[0].max().item():.3f}")
    print(f"  max|Δ|={((logits_A[0].float() - logits_Ap[0].float()).abs().max()).item():.4f}")

    # --------------------------------------------------------------------- (B'')
    # Localize: does adding EVEN ONE token after prefix perturb logit at pos P-1?
    # If yes → attention is not strictly causal in this forward path.
    print(f"\n[B''] prefix + 1 extra token; compare logit at pos P-1 ...")
    extra_token = seq[:, :P + 1]  # prefix + first gen token
    batch_B = {k: v for k, v in batch.items() if k != "attention_mask"}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm.model.rope_deltas = None
        out_Bpp = vlm(input_ids=extra_token, use_cache=False, **batch_B)
    logit_at_Pminus1_with_extra = out_Bpp.logits[0, P - 1]
    print(f"  with extra token: argmax@P-1 = {logit_at_Pminus1_with_extra.argmax().item()}  "
          f"top5 = {_top5(logit_at_Pminus1_with_extra).tolist()}")
    print(f"  A' (prefix only) argmax@P-1  = {logits_Ap[0].argmax().item()}  "
          f"top5 = {_top5(logits_Ap[0]).tolist()}")
    delta = (logit_at_Pminus1_with_extra.float() - logits_Ap[0].float()).abs().max().item()
    print(f"  max|Δ| at pos P-1: {delta:.4f}")

    # --------------------------------------------------------------------- (B)
    # Single-prefill: feed (prefix + N greedy tokens) in ONE forward, no past_kv.
    # Compare logits at positions P-1..P+N-2 (each predicts the next token).
    # Strip attention_mask (shape [P]); get_rope_index rebuilds ones_like when None.
    print(f"\n[B] single prefill over (prefix + {N} gen tokens) ...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm.model.rope_deltas = None
        out_B = vlm(
            input_ids=seq[:, :P + N],
            use_cache=False,
            **batch_B,
        )
    print(f"  out_B.logits.shape = {tuple(out_B.logits.shape)}")
    logits_full_B = out_B.logits[0]                  # (P+N, V)
    logits_B = logits_full_B[P - 1 : P - 1 + N]      # (N, V), predicts tokens P..P+N-1
    _compare("B: single-prefill vs (A) incremental", logits_A, logits_B)

    # --------------------------------------------------------------------- (C)
    # True verify: generate for 0 new tokens (just prefill the prefix to get cache),
    # then call model(N_tokens, past_kv=cache, cache_position=[P..P+N-1]).
    # This is exactly what a speculative verify forward looks like.
    print(f"\n[C] verify: prefill prefix → batched forward on {N} tokens w/ past_kv ...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm.model.rope_deltas = None
        prefill = vlm.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True,
            **batch,
        )
        # generate(max_new_tokens=1) runs ONE extra forward → cache now has P+1 positions.
        # Crop it back to just the prefix (length P) so the verify starts at position P.
        prompt_cache = prefill.past_key_values
        cache_len_before_crop = prompt_cache.get_seq_length()
        prompt_cache.crop(P)
        assert prompt_cache.get_seq_length() == P, \
            f"crop failed: cache len {prompt_cache.get_seq_length()} != {P}"
        # Also keep the rope_deltas from that prefill — HF generate() stored it on the model.
        rope_deltas = vlm.model.rope_deltas
        print(f"  prompt_cache len {cache_len_before_crop} → cropped to {P}; "
              f"rope_deltas = {rope_deltas.tolist() if rope_deltas is not None else None}")

        cache_position = torch.arange(P, P + N, device=device)
        out_C = vlm(
            input_ids=gen_tokens,
            past_key_values=prompt_cache,
            cache_position=cache_position,
            use_cache=True,
            # pass images so kwargs match A (vlm forward ignores them when past_kv is non-empty,
            # since inputs_embeds are built from input_ids without images on non-prefill path)
        )
    logits_C = out_C.logits[0]     # (N, V)
    print(f"  out_C.logits.shape = {tuple(out_C.logits.shape)}")
    # Off-by-one hypothesis: logits_C[j] predicts token P+j+1 (next after gen_tokens[j]),
    # while logits_A[i] predicts token P+i. So logits_C[j] ~ logits_A[j+1].
    # Compare logits_C[0..N-2] to logits_A[1..N-1].
    _compare("C: verify (shifted by -1) vs (A) [j: predicts A's i=j+1]",
             logits_A[1:N], logits_C[:N-1])

    # --------------------------------------------------------------------- summary
    print("\n=== INTERPRETATION ===")
    print("• If B matches A  →  Qwen3VL is consistent across batched-prefill vs ")
    print("                      incremental decode (expected).")
    print("• If C matches A  →  verify-style forward w/ past_kv produces correct ")
    print("                      logits. DFlash CAN be made to work: the previous ")
    print("                      0% accept was a bug in our old speculative.py, ")
    print("                      not a target-model property. → Phase 2: rebuild ")
    print("                      speculative.py, compare each piece to this script.")
    print("• If C diverges   →  position_ids / cache_position / rope_deltas wiring ")
    print("                      is broken even in HF's own forward path. Needs ")
    print("                      deeper fix (maybe MRoPE axis 0 only for text-beyond-")
    print("                      images isn't enough).")


if __name__ == "__main__":
    main()
