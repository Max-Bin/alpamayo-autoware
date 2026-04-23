from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort
import torch

from alpamayo1_5.trt.common import (
    flatten_past_key_values,
    legacy_cache_from_prompt_cache,
    past_key_value_input_names,
    resolve_artifact_dir,
    torch_dtype_to_onnxruntime_dtype,
)


@dataclass
class TrtExpertContext:
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    flat_past_key_values: list[torch.Tensor]


class TrtExpertEngine:
    """ONNX Runtime TensorRT wrapper for Alpamayo's expert denoiser."""

    def __init__(
        self,
        onnx_model_path: str | Path,
        engine_cache_dir: str | Path,
        enable_int8: bool = True,
        enable_fp16: bool = True,
    ) -> None:
        self.onnx_model_path = str(Path(onnx_model_path).expanduser().resolve())
        self.engine_cache_dir = resolve_artifact_dir(engine_cache_dir)
        self.enable_int8 = enable_int8
        self.enable_fp16 = enable_fp16

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        trt_provider_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(self.engine_cache_dir),
            "trt_fp16_enable": self.enable_fp16,
            "trt_int8_enable": self.enable_int8,
        }
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=session_options,
            providers=[
                ("TensorrtExecutionProvider", trt_provider_options),
                ("CUDAExecutionProvider", {}),
                ("CPUExecutionProvider", {}),
            ],
        )
        self.input_names = [entry.name for entry in self.session.get_inputs()]
        self.input_dtypes = {
            entry.name: self._tensor_dtype_from_onnx_type(entry.type)
            for entry in self.session.get_inputs()
        }
        self.output_name = self.session.get_outputs()[0].name
        self.output_dtype = self._output_dtype_from_onnx_type(self.session.get_outputs()[0].type)
        self.kv_input_names = [
            name
            for name in self.input_names
            if name.startswith("past_key_") or name.startswith("past_value_")
        ]

    def prepare_context(
        self,
        prompt_cache: object,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> TrtExpertContext:
        """Convert static prompt-side tensors into a cached TRT context."""
        legacy_cache = legacy_cache_from_prompt_cache(prompt_cache)
        flat_cache = []
        expected_kv_names = past_key_value_input_names(len(legacy_cache))
        for name, tensor in zip(expected_kv_names, flatten_past_key_values(legacy_cache)):
            flat_cache.append(
                tensor.detach().contiguous().to(
                    device=position_ids.device,
                    dtype=self.input_dtypes[name],
                )
            )
        return TrtExpertContext(
            position_ids=position_ids.detach().contiguous().to(dtype=self.input_dtypes["position_ids"]),
            attention_mask=attention_mask.detach().contiguous().to(
                dtype=self.input_dtypes["attention_mask"]
            ),
            flat_past_key_values=flat_cache,
        )

    def step(self, x: torch.Tensor, t: torch.Tensor, context: TrtExpertContext) -> torch.Tensor:
        """Run a single denoiser step using the TensorRT-backed ONNX graph."""
        device_id = x.device.index or 0
        output = torch.empty(tuple(x.shape), device=x.device, dtype=self.output_dtype)
        x_tensor = x.detach().contiguous().to(dtype=self.input_dtypes["x"])
        t_tensor = t.detach().contiguous().to(dtype=self.input_dtypes["t"])

        io_binding = self.session.io_binding()
        keepalive = [x_tensor, t_tensor, output, context.position_ids, context.attention_mask]
        self._bind_tensor(io_binding, "x", x_tensor, device_id)
        self._bind_tensor(io_binding, "t", t_tensor, device_id)
        self._bind_tensor(io_binding, "position_ids", context.position_ids, device_id)
        self._bind_tensor(io_binding, "attention_mask", context.attention_mask, device_id)

        expected_kv_names = past_key_value_input_names(len(context.flat_past_key_values) // 2)
        for name, tensor in zip(expected_kv_names, context.flat_past_key_values):
            keepalive.append(tensor)
            self._bind_tensor(io_binding, name, tensor, device_id)

        io_binding.bind_output(
            self.output_name,
            device_type="cuda",
            device_id=device_id,
            element_type=torch_dtype_to_onnxruntime_dtype(output.dtype),
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )
        io_binding.synchronize_inputs()
        self.session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        # Hold references until execution is complete.
        del keepalive
        return output

    @staticmethod
    def _bind_tensor(
        io_binding: ort.IOBinding,
        name: str,
        tensor: torch.Tensor,
        device_id: int,
    ) -> None:
        io_binding.bind_input(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=torch_dtype_to_onnxruntime_dtype(tensor.dtype),
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    @staticmethod
    def _output_dtype_from_onnx_type(type_name: str) -> torch.dtype:
        return TrtExpertEngine._tensor_dtype_from_onnx_type(type_name)

    @staticmethod
    def _tensor_dtype_from_onnx_type(type_name: str) -> torch.dtype:
        mapping = {
            "tensor(bool)": torch.bool,
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(int64)": torch.int64,
        }
        if type_name not in mapping:
            raise TypeError(f"Unsupported ONNX tensor type: {type_name}")
        return mapping[type_name]
