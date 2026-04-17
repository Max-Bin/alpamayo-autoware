from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from transformers.cache_utils import DynamicCache

from alpamayo_r1.trt.common import (
    build_dynamic_axes,
    flatten_past_key_values,
    legacy_cache_from_prompt_cache,
    past_key_value_input_names,
    resolve_artifact_dir,
)


class ExpertDenoiserExportModule(torch.nn.Module):
    """Wrap Alpamayo's expert denoiser step into a single exportable module."""

    def __init__(self, model: Any):
        super().__init__()
        self.action_in_proj = model.action_in_proj
        self.expert = model.expert
        self.action_out_proj = model.action_out_proj
        self.n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
        self.action_dims = tuple(model.action_space.get_action_space_dims())
        self.num_hidden_layers = int(self.expert.config.num_hidden_layers)
        self.is_causal = not bool(model.config.expert_non_causal_attention)
        try:
            self.action_in_proj_dtype = next(self.action_in_proj.parameters()).dtype
        except StopIteration:
            self.action_in_proj_dtype = torch.float32
        try:
            self.expert_dtype = next(self.expert.parameters()).dtype
        except StopIteration:
            self.expert_dtype = torch.float32

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        legacy_cache = []
        for idx in range(0, len(past_key_values), 2):
            past_key = past_key_values[idx].to(dtype=self.expert_dtype)
            past_value = past_key_values[idx + 1].to(dtype=self.expert_dtype)
            legacy_cache.append((past_key, past_value))
        prompt_cache = DynamicCache.from_legacy_cache(tuple(legacy_cache))

        future_token_embeds = self.action_in_proj(
            x.to(dtype=self.action_in_proj_dtype),
            t.to(dtype=self.action_in_proj_dtype),
        ).to(dtype=self.expert_dtype)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(batch_size, self.n_diffusion_tokens, -1)

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids.to(dtype=torch.int64),
            past_key_values=prompt_cache,
            attention_mask=attention_mask.to(dtype=self.expert_dtype),
            use_cache=False,
            is_causal=self.is_causal,
        )
        last_hidden = expert_out.last_hidden_state[:, -self.n_diffusion_tokens :]
        pred = self.action_out_proj(last_hidden).view(-1, *self.action_dims)
        return pred.to(dtype=self.expert_dtype)


class TensorFileCalibrationReader(CalibrationDataReader):
    """A lazy calibration reader that streams torch-saved tensor dicts from disk."""

    def __init__(self, sample_paths: list[Path]):
        self.sample_paths = sample_paths
        self._iterator = iter(self.sample_paths)

    def get_next(self) -> dict[str, Any] | None:
        try:
            sample_path = next(self._iterator)
        except StopIteration:
            return None
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        return {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()
        }

    def rewind(self) -> None:
        self._iterator = iter(self.sample_paths)


def load_calibration_sample(sample_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a captured denoiser sample to a target device."""
    sample = torch.load(Path(sample_path), map_location="cpu", weights_only=False)
    loaded: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            loaded[key] = value.to(device=device)
    return loaded


def export_expert_denoiser_onnx(
    model: Any,
    sample: dict[str, torch.Tensor],
    output_path: str | Path,
    opset_version: int = 18,
) -> Path:
    """Export Alpamayo's expert denoiser step to ONNX."""
    export_module = ExpertDenoiserExportModule(model).eval().to(device=sample["x"].device)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    past_key_values = legacy_cache_from_prompt_cache(
        tuple(
            (
                sample[f"past_key_{layer_idx:02d}"],
                sample[f"past_value_{layer_idx:02d}"],
            )
            for layer_idx in range(export_module.num_hidden_layers)
        )
    )
    flat_past = flatten_past_key_values(past_key_values)
    input_names = [
        "x",
        "t",
        "position_ids",
        "attention_mask",
        *past_key_value_input_names(export_module.num_hidden_layers),
    ]

    torch.onnx.export(
        export_module,
        (
            sample["x"],
            sample["t"],
            sample["position_ids"],
            sample["attention_mask"],
            *flat_past,
        ),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=["pred"],
        dynamic_axes=build_dynamic_axes(export_module.num_hidden_layers),
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        external_data=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(output_path))
    return output_path


def discover_nodes_to_exclude(model_path: str | Path) -> list[str]:
    """Exclude small projection heads from INT8 quantization for better fidelity."""
    model = onnx.load(Path(model_path), load_external_data=False)
    excluded_prefixes = ("action_in_proj", "action_out_proj")
    excluded = []
    for node in model.graph.node:
        if any(prefix in node.name for prefix in excluded_prefixes):
            excluded.append(node.name)
    return excluded


def apply_smoothquant_to_expert_denoiser(
    source_model_path: str | Path,
    output_model_path: str | Path,
    calibration_sample_paths: list[Path],
    smoothquant_alpha: float,
    execution_provider: str,
) -> tuple[Path, list[str]]:
    """Apply SmoothQuant with an explicit ORT execution provider."""
    try:
        importlib.import_module("neural_compressor.adaptor.ox_utils.smooth_quant")
    except Exception as exc:
        raise RuntimeError(
            "neural-compressor is not correctly installed. Please check your environment."
        ) from exc

    from neural_compressor.adaptor.ox_utils.smooth_quant import ORTSmoothQuant

    source_model_path = Path(source_model_path).expanduser().resolve()
    output_model_path = Path(output_model_path).expanduser().resolve()
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    original_model = onnx.load(source_model_path, load_external_data=False)
    original_node_names = {node.name for node in original_model.graph.node}

    def dataloader():
        reader = TensorFileCalibrationReader(calibration_sample_paths)
        for sample in reader:
            yield sample, None

    smoothquant = ORTSmoothQuant(
        str(source_model_path),
        dataloader(),
        reduce_range=False,
        backend=execution_provider,
    )
    smoothed_model = smoothquant.transform(
        alpha=smoothquant_alpha,
        folding=True,
        calib_iter=max(1, len(calibration_sample_paths)),
    )
    smoothed_model.save(str(output_model_path))

    smoothed_node_names = {node.name for node in smoothed_model.model.graph.node}
    inserted_nodes = sorted(smoothed_node_names - original_node_names)
    return output_model_path, inserted_nodes


def quantize_expert_denoiser_int8(
    source_model_path: str | Path,
    int8_model_path: str | Path,
    calibration_sample_paths: list[Path],
    calibration_method: CalibrationMethod = CalibrationMethod.Entropy,
    smoothquant_alpha: float = 0.6,
    smoothquant_provider: str = "CUDAExecutionProvider",
    calibration_providers: list[str] | None = None,
) -> Path:
    """Apply selective INT8 QDQ quantization to the exported denoiser graph."""
    reader = TensorFileCalibrationReader(calibration_sample_paths)
    nodes_to_exclude = set(discover_nodes_to_exclude(source_model_path))
    calibration_providers = calibration_providers or [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    common_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "CalibTensorRangeSymmetric": True,
        "DedicatedQDQPair": True,
        "MatMulConstBOnly": True,
        "CalibMaxIntermediateOutputs": 32,
    }

    with tempfile.TemporaryDirectory(prefix="alpamayo.smoothquant.") as temp_dir:
        smoothquant_model_path, inserted_nodes = apply_smoothquant_to_expert_denoiser(
            source_model_path=source_model_path,
            output_model_path=Path(temp_dir) / "expert_step.smoothquant.onnx",
            calibration_sample_paths=calibration_sample_paths,
            smoothquant_alpha=smoothquant_alpha,
            execution_provider=smoothquant_provider,
        )
        nodes_to_exclude.update(inserted_nodes)

        quantize_static(
            model_input=str(smoothquant_model_path),
            model_output=str(int8_model_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=sorted(nodes_to_exclude),
            use_external_data_format=True,
            calibrate_method=calibration_method,
            calibration_providers=calibration_providers,
            extra_options=common_options,
        )
    return Path(int8_model_path).expanduser().resolve()


def initialize_artifact_layout(output_dir: str | Path) -> dict[str, Path]:
    """Create a conventional artifact layout for TRT export outputs."""
    artifact_dir = resolve_artifact_dir(output_dir)
    layout = {
        "artifact_dir": artifact_dir,
        "calibration_dir": artifact_dir / "calibration",
        "engine_cache_dir": artifact_dir / "engine_cache",
        "fp32_onnx": artifact_dir / "expert_step.fp32.onnx",
        "int8_onnx": artifact_dir / "expert_step.int8.qdq.onnx",
        "manifest": artifact_dir / "manifest.json",
    }
    layout["calibration_dir"].mkdir(parents=True, exist_ok=True)
    layout["engine_cache_dir"].mkdir(parents=True, exist_ok=True)
    return layout
