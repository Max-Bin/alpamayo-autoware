#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from onnxruntime.quantization import CalibrationMethod

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.trt.common import flatten_past_key_values, legacy_cache_from_prompt_cache
from alpamayo1_5.trt.expert_runtime import TrtExpertEngine
from alpamayo1_5.trt.export import (
    export_expert_denoiser_onnx,
    initialize_artifact_layout,
    load_calibration_sample,
    quantize_expert_denoiser_int8,
)


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000
DEFAULT_OUTPUT_DIR = "~/autoware_data/alpamayo/v0.1"


class CalibrationCollector:
    """Capture denoiser step inputs and write them as torch tensors to disk."""

    def __init__(self, calibration_dir: Path, max_samples: int):
        self.calibration_dir = calibration_dir
        self.max_samples = max_samples
        self.sample_paths: list[Path] = []

    def __call__(self, step_inputs: dict[str, Any]) -> None:
        if len(self.sample_paths) >= self.max_samples:
            return

        prompt_cache = legacy_cache_from_prompt_cache(step_inputs["prompt_cache"])
        sample: dict[str, torch.Tensor] = {
            "x": step_inputs["x"].detach().to(dtype=torch.float16).cpu(),
            "t": step_inputs["t"].detach().to(dtype=torch.float16).cpu(),
            "position_ids": step_inputs["position_ids"].detach().cpu(),
            "attention_mask": step_inputs["attention_mask"].detach().cpu(),
        }
        for layer_idx, tensor in enumerate(flatten_past_key_values(prompt_cache)):
            name = (
                f"past_key_{layer_idx // 2:02d}"
                if layer_idx % 2 == 0
                else f"past_value_{layer_idx // 2:02d}"
            )
            sample[name] = tensor.detach().to(dtype=torch.float16).cpu()

        output_path = self.calibration_dir / f"sample_{len(self.sample_paths):03d}.pt"
        torch.save(sample, output_path)
        self.sample_paths.append(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a split TensorRT engine for Alpamayo's expert denoiser."
    )
    parser.add_argument("--model-id", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
    parser.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    parser.add_argument("--max-generation-length", type=int, default=64)
    parser.add_argument("--num-calibration-samples", type=int, default=8)
    parser.add_argument(
        "--calibration-method",
        choices=("entropy", "percentile", "minmax"),
        default="entropy",
    )
    parser.add_argument("--smoothquant-alpha", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def calibration_method_from_name(name: str) -> CalibrationMethod:
    mapping = {
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "minmax": CalibrationMethod.MinMax,
    }
    return mapping[name]


def prepare_model_inputs(model: Alpamayo1_5, clip_id: str, t0_us: int) -> tuple[dict[str, Any], dict[str, Any]]:
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, "cuda"), data


def run_full_inference(
    model: Alpamayo1_5,
    model_inputs: dict[str, Any],
    max_generation_length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inference_inputs = copy.deepcopy(model_inputs)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inference_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            num_traj_sets=1,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    return pred_xyz.detach().cpu(), pred_rot.detach().cpu(), extra


def main() -> None:
    args = parse_args()
    layout = initialize_artifact_layout(args.output_dir)

    model = Alpamayo1_5.from_pretrained(args.model_id, dtype=torch.bfloat16).to("cuda")
    model.eval()
    model_inputs, _ = prepare_model_inputs(model, args.clip_id, args.t0_us)

    collector = CalibrationCollector(layout["calibration_dir"], args.num_calibration_samples)
    model.set_expert_step_observer(collector)
    run_full_inference(model, model_inputs, args.max_generation_length, args.seed)
    model.set_expert_step_observer(None)
    if not collector.sample_paths:
        raise RuntimeError("No denoiser steps were captured for calibration.")

    export_sample = load_calibration_sample(collector.sample_paths[0], device=torch.device("cuda"))
    model.action_in_proj = model.action_in_proj.to(device="cuda", dtype=torch.float32)
    model.expert = model.expert.to(device="cuda", dtype=torch.float32)
    model.action_out_proj = model.action_out_proj.to(device="cuda", dtype=torch.float32)
    export_expert_denoiser_onnx(model, export_sample, layout["fp32_onnx"])
    quantize_expert_denoiser_int8(
        source_model_path=layout["fp32_onnx"],
        int8_model_path=layout["int8_onnx"],
        calibration_sample_paths=collector.sample_paths,
        calibration_method=calibration_method_from_name(args.calibration_method),
        smoothquant_alpha=args.smoothquant_alpha,
        smoothquant_provider="CUDAExecutionProvider",
        calibration_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    onnx_model_path = layout["int8_onnx"]
    trt_engine = TrtExpertEngine(
        onnx_model_path=onnx_model_path,
        engine_cache_dir=layout["engine_cache_dir"],
        enable_int8=True,
        enable_fp16=True,
    )
    trt_context = trt_engine.prepare_context(
        prompt_cache=tuple(
            (
                export_sample[f"past_key_{layer_idx:02d}"].to(device="cuda"),
                export_sample[f"past_value_{layer_idx:02d}"].to(device="cuda"),
            )
            for layer_idx in range(model.expert.config.num_hidden_layers)
        ),
        position_ids=export_sample["position_ids"].to(device="cuda"),
        attention_mask=export_sample["attention_mask"].to(device="cuda"),
    )
    _ = trt_engine.step(
        x=export_sample["x"].to(device="cuda"),
        t=export_sample["t"].to(device="cuda"),
        context=trt_context,
    )

    manifest = {
        "model_id": args.model_id,
        "clip_id": args.clip_id,
        "t0_us": args.t0_us,
        "seed": args.seed,
        "fp32_onnx": str(layout["fp32_onnx"]),
        "int8_onnx": str(layout["int8_onnx"]),
        "engine_cache_dir": str(layout["engine_cache_dir"]),
        "calibration_method": args.calibration_method,
        "smoothquant_alpha": args.smoothquant_alpha,
        "num_calibration_samples": len(collector.sample_paths),
    }

    if not args.skip_validation:
        del model
        torch.cuda.empty_cache()
        model = Alpamayo1_5.from_pretrained(args.model_id, dtype=torch.bfloat16).to("cuda")
        model.eval()
        model_inputs, _ = prepare_model_inputs(model, args.clip_id, args.t0_us)

        native_pred_xyz, native_pred_rot, native_extra = run_full_inference(
            model=model,
            model_inputs=model_inputs,
            max_generation_length=args.max_generation_length,
            seed=args.seed,
        )
        model.set_expert_step_runner(trt_engine)
        trt_pred_xyz, trt_pred_rot, trt_extra = run_full_inference(
            model=model,
            model_inputs=model_inputs,
            max_generation_length=args.max_generation_length,
            seed=args.seed,
        )

        traj_abs_diff = (native_pred_xyz - trt_pred_xyz).abs()
        rot_abs_diff = (native_pred_rot - trt_pred_rot).abs()
        manifest["validation"] = {
            "trajectory_max_abs_diff": float(traj_abs_diff.max().item()),
            "trajectory_mean_abs_diff": float(traj_abs_diff.mean().item()),
            "rotation_max_abs_diff": float(rot_abs_diff.max().item()),
            "rotation_mean_abs_diff": float(rot_abs_diff.mean().item()),
            "native_cot": str(native_extra["cot"][0, 0, 0]),
            "trt_cot": str(trt_extra["cot"][0, 0, 0]),
        }

    layout["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
