# Experiments

This file catalogs major optimisation attempts that were **investigated in depth
but not shipped** on the main `alpamayo1.5-flashdrive` branch. The full commits
live on dedicated `experiments/*` branches so the infrastructure can be picked
up again without re-discovering the failure modes.

---

## 1. DFlash block-diffusion speculative decoding — not shipped

- **Branch with full implementation:** `experiments/dflash-attempt` (this branch)
- **Draft checkpoint:** `FlashDriveVLA/Alpamayo-1.5-DFlash` on HuggingFace

### Motivation

PARO W4A8 (see §2) only delivers paper-level speedup on the *decode* path when
the per-step M rises above vLLM Marlin's break-even point (≥ 16). Without
speculative decoding, our streaming inference runs M=1 per token, so decode
falls back to a cached BF16 dequant (no weight-read speedup). DFlash is the
paired speculative decoder designed to push M per step into the W4A8-friendly
range by proposing 8 draft tokens and verifying in one batched target forward.
If it worked we'd expect mean_accept ≈ 4–5 and ≈ 2× e2e decode speedup.

### What worked

- **Target-model consistency** (`scripts/debug_dflash_step.py`):
  proved that Qwen3VL's batched-verify forward (past_kv populated,
  `cache_position=[P..P+N-1]`, no explicit `position_ids`) produces logits
  bit-identical to HF's incremental decode. 7/7 argmax match, max|Δ|=0.25,
  mean KL=0.0001 on real rosbag input. The **shift-by-one** alignment:
  `out.logits[:, j]` predicts the token at position `P+j+1` (not `P+j`).
- **VLM-aware `dflash_generate`** (`src/alpamayo1_5/flashdrive/dflash_spec.py`):
  handles the Qwen3VL target's 3-axis MRoPE (via rope_deltas + cache_position),
  the OOV `mask_token_id` (`155697` lives outside the embed table — spliced
  in via the shipped `mask_embedding.pt` at masked positions), and
  rope-delta-shifted draft position_ids (draft is plain Qwen3 trained on
  text-scale positions, so the absolute prefix positions 3073+ would be OOD).
- **Correctness end-to-end:** greedy output from DFlash matches HF greedy
  baseline **16/16 tokens**. The verify loop, cache handling, position
  wiring are all correct.

### What failed

    baseline HF greedy  :  534 ms
    DFlash spec-decode  :  741 ms   ← slower, not faster
    mean_accept         :  1.07     ← draft accepts ~0 tokens per block
    greedy-agreement    :  16/16    ← output correctness ✓

The draft mode-collapses: regardless of context, it predicts one of a tiny set
of common filler tokens (279, 10497, 21272, 5828). Inspecting draft weights
(all norms reasonable) and the mask_embedding buffer (norm=0.1 — tiny but
consistent with "null signal" training) rules out a load bug.

### Root cause

**Not an integration bug — a draft/target distribution mismatch.** The
released `FlashDriveVLA/Alpamayo-*-DFlash` drafts were trained against
**sampling + no trajectory-token mask**. Our production inference path runs
**greedy + ExpertLogitsProcessor** (which zeros out the ~4000 trajectory
tokens). The draft's token distribution is out-of-distribution for the
target argmax at every position. Switching to `temperature=0.6`/`top_p=0.98`
to match training did **not** move `mean_accept` above 1.14.

### Paths forward (if revisited)

1. **Retrain the draft** for our greedy + traj-mask inference regime
   (GPU-weeks; requires the training recipe Z Lab has not open-sourced yet
   per the DFlash README).
2. **Medusa / EAGLE-style draft heads** trained alongside our model — no
   paired pre-trained checkpoint needed, but still a training project.
3. **n-gram / PLD speculative** — cheap, requires no training; accept rate
   will be lower than a learned draft but implementation is trivial.
4. **Route through vLLM's native DFlash backend** — vLLM has first-class
   support (DFlash README shows `vllm serve ... --speculative-config '{"method":
   "dflash", ...}'`) — requires moving the VLM inference path off transformers
   onto vLLM, a substantial rewrite.

### How to pick this back up

```bash
git checkout experiments/dflash-attempt
PYTHONPATH=src python scripts/debug_dflash_step.py   # Phase-1 consistency check
PYTHONPATH=src python scripts/bench_dflash.py        # end-to-end mean_accept bench
```

Edit `src/alpamayo1_5/flashdrive/dflash_spec.py` to swap in a different draft
(Medusa head, n-gram table, ...) — the verify loop is target-agnostic.

---

## 2. PARO W4A8 quantisation — not shipped

- **Branch with full implementation:** `experiments/paro-attempt`
- **Checkpoint:** `FlashDriveVLA/Alpamayo-1.5-10B-finetuned-PARO` on HuggingFace

### Measured result

**+0.13× net** on top of the rest of the FlashDrive stack (2.02× → 2.16× p50
on the streaming-rosbag bench).

### Why the gain is smaller than the paper suggests

- The paper's PARO decode win hinges on **DFlash lifting per-step M ≥ 16** so
  Marlin W4A8 kernels actually run. Without DFlash (see §1), our decode
  path stays at M=1 per token → Marlin's 36 µs/call overhead kills the W4
  advantage → we fall back to cached BF16 dequant which has zero speedup over
  baseline BF16 GEMV. That zeroes ≈ 25% of e2e time (decode steps) to PARO's
  contribution.
- **`streaming_lm`** (pre-RoPE K/V + hidden_out cache + MLP slice) already
  eliminates most prefill weight-read cost in steady state. PARO's prefill win
  now has a much smaller absolute-millisecond budget to carve from.
- We only quantize the 252 LLM Linears that the PARO checkpoint declares.
  ViT, action expert, patch_embed stay BF16.

### Why it's not on main

+0.13× does not justify the maintenance surface: `MarlinRotatedLinear`,
hybrid M<16 / M≥16 dispatch, FP16↔BF16 cast management around the Marlin
kernel, weight-layout workaround for cuBLAS (`.t().contiguous()` → 3.6× GEMV
at M=1), separate checkpoint to download, and an order-dependent interaction
with `kernel_fusion` (PARO must install first, then kernel_fusion's
`isinstance(..., nn.Linear)` check silently skips the already-swapped
`MarlinRotatedLinear` layers).

### How to pick this back up

```bash
git checkout experiments/paro-attempt
huggingface-cli download FlashDriveVLA/Alpamayo-1.5-10B-finetuned-PARO \
  --local-dir models/Alpamayo-1.5-10B-finetuned-PARO
PYTHONPATH=src python scripts/bench_paro_combined.py \
  --teacher models/Alpamayo-1.5-10B-finetuned \
  --paro    models/Alpamayo-1.5-10B-finetuned-PARO \
  --bag     /path/to/rosbag --steps 8
```

This becomes newly attractive the moment a working spec-decode path exists
(Medusa heads trained in-house, vLLM-native, ...) because then PARO's
decode-phase contribution finally materialises.

---

## Conventions for future entries

- Put the full implementation behind an `experiments/<name>` branch so git gc
  won't reclaim it and `git checkout` restores a runnable state.
- Keep the paired checkpoint on the HuggingFace org; don't duplicate weights
  into git.
- Record in this file: motivation, what worked, what failed, *measured*
  numbers (not theorised), the root cause if diagnosed, and concrete paths
  forward.
- Cite the reproducing script(s) so anyone can re-run in five minutes.
