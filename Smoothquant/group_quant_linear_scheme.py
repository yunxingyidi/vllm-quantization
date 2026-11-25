from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any

import torch
import torch.nn.functional as F

# from Smoothquant.GroupQuant.group_quant_npu import fake_quantize_activation_per_token
from vllm.logger import init_logger
try:
    from vllm_ascend.quantization.methods.base import AscendLinearScheme
except ModuleNotFoundError:
    class AscendLinearScheme:  # type: ignore[override]
        pass

if TYPE_CHECKING:
    from Smoothquant.group_quant_config import GroupQuantConfig


logger = init_logger(__name__)


@torch.no_grad()
def quantize_weight_per_group_absmax(
    weight: torch.Tensor,
    n_bits: int = 8,
    residual_bits: int = 4,
    residual_group: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    if in_features % residual_group != 0:
        raise ValueError(
            f"Input size {in_features} must be divisible by residual_group "
            f"{residual_group}."
        )

    q_max = 2 ** (n_bits - 1) - 1
    e_min = -(2 ** (residual_bits - 1))
    n_groups = in_features // residual_group

    weight_group = weight.reshape(out_features, n_groups, residual_group)
    delta = weight_group.abs().amax(dim=-1).clamp_(min=1e-5) / q_max
    delta_base = delta.amax(dim=-1)

    r = (delta_base / (1 << residual_bits)).unsqueeze(-1)
    safe_r = r.clamp_min(1e-5)

    e = torch.round((delta - delta_base.unsqueeze(-1)) / safe_r)
    e = e.clamp_(min=e_min, max=0).to(torch.int8)

    delta_rec = delta_base.unsqueeze(-1) + e.to(delta.dtype) * r
    delta_rec = delta_rec.clamp_(min=1e-5)

    weight_q = torch.round(weight_group / delta_rec.unsqueeze(-1))
    weight_q = weight_q.clamp_(-q_max, q_max).to(torch.int8)
    return weight_q.reshape_as(weight), delta_base.to(torch.bfloat16), e


@torch.no_grad()
def dequantize_weight_per_group_absmax(
    weight_q: torch.Tensor,
    delta_base: torch.Tensor,
    e: torch.Tensor,
    residual_bits: int = 4,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if weight_q.dim() != 2:
        raise ValueError(
            f"Expected 2D quantized weight tensor, got {tuple(weight_q.shape)}"
        )

    out_features, in_features = weight_q.shape
    n_groups = e.shape[-1]
    residual_group = in_features // n_groups

    weight_group = weight_q.reshape(out_features, n_groups, residual_group).to(
        torch.float32
    )
    delta_base_fp = delta_base.to(torch.float32).unsqueeze(-1)
    r = delta_base_fp / (1 << residual_bits)
    delta_rec = (delta_base_fp + e.to(torch.float32) * r).clamp_(min=1e-5)

    return (weight_group * delta_rec.unsqueeze(-1)).reshape_as(weight_q).to(output_dtype)


@torch.no_grad()
def quantize_weight_last_dim_runtime(
    weight: torch.Tensor,
    n_bits: int,
    residual_bits: int,
    residual_group: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trailing_size = weight.shape[-1]
    if trailing_size % residual_group != 0:
        raise ValueError(
            f"Last dim {trailing_size} must be divisible by residual_group "
            f"{residual_group}."
        )

    reshaped = weight.reshape(-1, trailing_size)
    weight_q, delta_base, e = quantize_weight_per_group_absmax(
        weight=reshaped,
        n_bits=n_bits,
        residual_bits=residual_bits,
        residual_group=residual_group,
    )
    return (
        weight_q.reshape_as(weight),
        delta_base.reshape(weight.shape[:-1]).contiguous(),
        e.reshape(*weight.shape[:-1], -1).contiguous(),
    )


@torch.no_grad()
def dequantize_weight_last_dim_runtime(
    weight_q: torch.Tensor,
    delta_base: torch.Tensor,
    e: torch.Tensor,
    residual_bits: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    trailing_size = weight_q.shape[-1]
    reshaped_q = weight_q.reshape(-1, trailing_size)
    reshaped_delta = delta_base.reshape(-1)
    reshaped_e = e.reshape(-1, e.shape[-1])
    weight = dequantize_weight_per_group_absmax(
        weight_q=reshaped_q,
        delta_base=reshaped_delta,
        e=reshaped_e,
        residual_bits=residual_bits,
        output_dtype=output_dtype,
    )
    return weight.reshape_as(weight_q)


def fake_quantize_weight_last_dim(
    weight: torch.Tensor,
    n_bits: int,
    residual_bits: int,
    residual_group: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output_dtype = output_dtype or weight.dtype
    weight_q, delta_base, e = quantize_weight_last_dim_runtime(
        weight=weight,
        n_bits=n_bits,
        residual_bits=residual_bits,
        residual_group=residual_group,
    )
    return dequantize_weight_last_dim_runtime(
        weight_q=weight_q,
        delta_base=delta_base,
        e=e,
        residual_bits=residual_bits,
        output_dtype=output_dtype,
    )


@torch.no_grad()
def fake_quantize_activation_last_dim(
    x: torch.Tensor,
    n_bits: int,
    residual_bits: int,
    residual_group: int,
) -> torch.Tensor:
    """Group-quantize/dequantize each token along its last dimension."""
    hidden_size = x.shape[-1]
    if hidden_size % residual_group != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by residual_group "
            f"{residual_group}."
        )
    flat = x.reshape(-1, hidden_size)
    activation_q, delta_base, e = quantize_weight_per_group_absmax(
        weight=flat,
        n_bits=n_bits,
        residual_bits=residual_bits,
        residual_group=residual_group,
    )
    dequantized = dequantize_weight_per_group_absmax(
        weight_q=activation_q,
        delta_base=delta_base,
        e=e,
        residual_bits=residual_bits,
        output_dtype=x.dtype,
    )
    return dequantized.reshape_as(x)


class GroupQuantLinearScheme(AscendLinearScheme):
    """Linear runtime for GroupQuant with FP16 fallback matmul.

    Runtime contract:
    - `weight_q`, `weight_delta_base`, and `weight_e` are the primary low-bit
      execution state.
    - grouped scales are reconstructed from `delta_base/e` inside `apply()` or
      inside a future fused kernel path.
    """

    def __init__(self, quant_config: "GroupQuantConfig"):
        self.quant_config = quant_config

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "weight": torch.empty(output_size, input_size, dtype=params_dtype),
        }

    def get_pergroup_param(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_type: str | None = None,
    ) -> dict[str, Any]:
        n_groups = input_size // self.quant_config.residual_group
        return {
            "weight_q": torch.empty(output_size, input_size, dtype=torch.int8),
            "weight_delta_base": torch.empty(output_size, dtype=torch.bfloat16),
            "weight_e": torch.empty(output_size, n_groups, dtype=torch.int8),
        }

    def setup_layer(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
    ) -> None:
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        layer.group_quant_runtime = "fp16_fallback"
        layer.group_quant_has_int8_kernel = (
            os.getenv("GROUP_QUANT_USE_QUANT_GROUP_MATMUL", "0") == "1"
            and self.quant_config.quantize_activations
            and self.quant_config.act_bits == 8
        )
        layer.group_quant_weight_ready = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self.quant_config.quantize_weights:
            layer.group_quant_weight_ready = False
            return

        try:
            weight_q, delta_base, e = quantize_weight_per_group_absmax(
                weight=layer.weight.data,
                n_bits=self.quant_config.weight_bits,
                residual_bits=self.quant_config.residual_bits,
                residual_group=self.quant_config.residual_group,
            )
            layer.weight_q.data.copy_(weight_q.contiguous())
            layer.weight_delta_base.data.copy_(delta_base.contiguous())
            layer.weight_e.data.copy_(e.contiguous())
            layer.group_quant_weight_ready = True
        except ValueError as exc:
            logger.warning_once(
                "Skipping GroupQuant weight quantization for %s: %s",
                getattr(layer, "prefix", layer.__class__.__name__),
                exc,
            )
            layer.group_quant_weight_ready = False

    def prepare_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_config.quantize_activations:
            return x

        try:
            return fake_quantize_activation_last_dim(
                x,
                n_bits=self.quant_config.act_bits,
                residual_bits=self.quant_config.residual_bits,
                residual_group=self.quant_config.residual_group,
            )
        except ValueError as exc:
            logger.warning_once(
                "Skipping GroupQuant activation quantization for %s: %s",
                getattr(layer, "prefix", layer.__class__.__name__),
                exc,
            )
            return x

    def get_fallback_weight(
        self,
        layer: torch.nn.Module,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if getattr(layer, "group_quant_weight_ready", False):
            return dequantize_weight_per_group_absmax(
                weight_q=layer.weight_q,
                delta_base=layer.weight_delta_base,
                e=layer.weight_e,
                residual_bits=self.quant_config.residual_bits,
                output_dtype=output_dtype,
            )
        return layer.weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        x = self.prepare_activation(layer, x)
        weight = self.get_fallback_weight(layer, x.dtype)
        return F.linear(x, weight, bias)
