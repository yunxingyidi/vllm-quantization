from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from Smoothquant.group_quant_linear_scheme import (
    GroupQuantLinearScheme,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from Smoothquant.group_quant_config import GroupQuantConfig


logger = init_logger(__name__)
_PROFILE_TOTALS: dict[str, float] = {}
_PROFILE_CALLS = 0


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning_once("Ignoring invalid %s=%r; using %d.", name, value, default)
        return default
    if parsed <= 0:
        logger.warning_once("Ignoring non-positive %s=%r; using %d.", name, value, default)
        return default
    return parsed


def _maybe_log_nonfinite(name: str, tensor: torch.Tensor) -> bool:
    if os.getenv("GROUP_QUANT_CHECK_NONFINITE", "0") != "1":
        return False
    if not torch.is_floating_point(tensor):
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if not has_nan and not has_inf:
        return False
    logger.error(
        "GroupQuant non-finite tensor detected: %s shape=%s dtype=%s nan=%s inf=%s",
        name,
        tuple(tensor.shape),
        tensor.dtype,
        has_nan,
        has_inf,
    )
    return True


def _group_weight_scale(
    delta_base: torch.Tensor,
    e: torch.Tensor,
    residual_bits: int,
    use_moe = False
) -> torch.Tensor:
    if use_moe:
        delta_base = delta_base.unsqueeze(-1)
    if delta_base.dim() == 1 and e.dim() == 2:
        delta_base = delta_base.unsqueeze(-1)
    delta_base_bf16 = delta_base.to(torch.bfloat16)
    return (
        delta_base_bf16
        + e.to(torch.bfloat16) * (delta_base_bf16 / (1 << residual_bits))
    ).clamp_min_(1e-5)


def _pack_grouped_weight_to_nz(
    weight_q: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    output_size, hidden_size = weight_q.shape
    if hidden_size % group_size != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by group_size {group_size}."
        )
    if group_size % 16 != 0:
        raise ValueError(f"group_size {group_size} must be divisible by 16 for NZ pack.")
    if output_size % 32 != 0:
        raise ValueError(f"Output size {output_size} must be divisible by 32 for NZ pack.")

    num_groups = hidden_size // group_size
    grouped = weight_q.view(output_size, num_groups, group_size).permute(1, 2, 0).contiguous()
    packed = grouped.view(num_groups, group_size // 16, 16, output_size // 32, 32)
    packed = packed.permute(0, 3, 1, 2, 4).contiguous()
    return packed.reshape(-1)


def _pack_grouped_activation_q(
    x_q: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    num_tokens, hidden_size = x_q.shape
    if hidden_size % group_size != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by group_size {group_size}."
        )
    num_groups = hidden_size // group_size
    if num_groups == 1:
        # Full-K grouping is already laid out as the single [M, K] matrix the
        # expert kernel consumes.  Avoid materializing a degenerate
        # view->permute->contiguous path (and the resulting Transpose launch).
        return x_q.contiguous().view(num_tokens, group_size)
    # Kernel group_axis=1 consumes K-groups as independent contiguous [M, G]
    # matrices laid out in GM one-after-another.
    return (
        x_q.view(num_tokens, num_groups, group_size)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(num_groups * num_tokens, group_size)
    )


def _dynamic_quant_by_group(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    num_tokens, hidden_size = x.shape
    if hidden_size % group_size != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by group_size {group_size}."
        )
    if os.getenv("GROUP_QUANT_USE_FUSED_DYNAMIC_QUANT", "1") == "1":
        try:
            from custom_op.quant_group_matmul.python_api import dynamic_quant_k_grouped

            return dynamic_quant_k_grouped(x, group_size)
        except Exception as exc:
            logger.warning_once(
                "Fused GroupQuant dynamic quant prototype failed: %s; "
                "falling back to torch_npu.npu_dynamic_quant.",
                exc,
            )

    num_groups = hidden_size // group_size
    x_grouped = (
        x.view(num_tokens, num_groups, group_size)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(num_groups * num_tokens, group_size)
    )
    x_q_packed, x_scale_packed = torch_npu.npu_dynamic_quant(x_grouped)
    return x_q_packed, x_scale_packed.view(num_groups, num_tokens).contiguous()


def _set_nonpersistent_buffer(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    if name in layer._buffers:
        layer._buffers[name] = value
    else:
        layer.register_buffer(name, value, persistent=False)


def _profile_sync_time(label: str, start: float | None = None) -> float:
    if start is None:
        torch.npu.synchronize()
        return time.perf_counter()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    _PROFILE_TOTALS[label] = _PROFILE_TOTALS.get(label, 0.0) + elapsed
    return time.perf_counter()


def _maybe_log_profile() -> None:
    global _PROFILE_CALLS
    _PROFILE_CALLS += 1
    if _PROFILE_CALLS % int(os.getenv("GROUP_QUANT_PROFILE_EVERY", "200")) != 0:
        return
    summary = ", ".join(
        f"{name}={value / _PROFILE_CALLS * 1000:.3f}ms"
        for name, value in sorted(_PROFILE_TOTALS.items())
    )
    logger.warning("GroupQuant fused profile avg over %d calls: %s", _PROFILE_CALLS, summary)


class GroupQuantLinearMethod(UnquantizedLinearMethod):
    """GroupQuant linear backend with dynamic activation quantization."""

    def __init__(self, quant_config: "GroupQuantConfig"):
        super().__init__()
        self.quant_config = quant_config
        self.scheme = GroupQuantLinearScheme(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.pop("weight_loader")

        self.scheme.setup_layer(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            params_dtype=params_dtype,
        )

        weight = ModelWeightParameter(
            data=self.scheme.get_weight(
                input_size=input_size_per_partition,
                output_size=output_size_per_partition,
                params_dtype=params_dtype,
            )["weight"],
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        for name, data in self.scheme.get_pergroup_param(
            input_size=input_size_per_partition,
            output_size=output_size_per_partition,
            params_dtype=params_dtype,
        ).items():
            param = torch.nn.Parameter(data, requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        self.scheme.process_weights_after_loading(layer)
        if not getattr(layer, "group_quant_weight_ready", False):
            return

        group_size = self.quant_config.residual_group
        try:
            weight_scale_grouped = _group_weight_scale(
                delta_base=layer.weight_delta_base,
                e=layer.weight_e,
                residual_bits=self.quant_config.residual_bits,
            ).transpose(0, 1).contiguous().to(torch.float32)
            _set_nonpersistent_buffer(
                layer,
                "group_quant_weight_scale_fused",
                weight_scale_grouped,
            )

            if os.getenv("GROUP_QUANT_CACHE_WEIGHT_NZ", "0") == "1":
                _set_nonpersistent_buffer(
                    layer,
                    "group_quant_weight_nz_fused",
                    _pack_grouped_weight_to_nz(layer.weight_q, group_size),
                )
        except ValueError as exc:
            logger.warning_once(
                "Skipping fused GroupQuant cache for %s: %s",
                getattr(layer, "prefix", layer.__class__.__name__),
                exc,
            )

        # The checkpoint BF16 weight is only a source for online quantization.
        # Runtime fallback reconstructs from weight_q/delta_base/e, so retaining
        # it would double the dominant model-weight allocation on every rank.
        if "weight" in layer._parameters:
            del layer.weight

    def _apply_fp16_fallback(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.scheme.get_fallback_weight(layer, x.dtype)
        return F.linear(x, weight, bias)

    def _apply_int8_kernel(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        try:
            import torch_npu
        except ModuleNotFoundError:
            logger.warning_once(
                "torch_npu is not available; falling back to FP16 linear for %s.",
                getattr(layer, "prefix", layer.__class__.__name__),
            )
            return self._apply_fp16_fallback(layer, x, bias)

        if not getattr(layer, "group_quant_weight_ready", False):
            return self._apply_fp16_fallback(layer, x, bias)

        input_shape = x.shape
        hidden_size = input_shape[-1]
        group_size = self.quant_config.residual_group
        if hidden_size % group_size != 0:
            logger.warning_once(
                "Hidden size %s is not divisible by residual_group %s for %s; "
                "falling back to FP16.",
                hidden_size,
                group_size,
                getattr(layer, "prefix", layer.__class__.__name__),
            )
            return self._apply_fp16_fallback(layer, x, bias)

        if os.getenv("GROUP_QUANT_USE_QUANT_GROUP_MATMUL", "0") == "1":
            out = self._apply_quant_group_matmul_fused(layer, x)
            if bias is not None:
                out = out + bias
            return out

        flat_x = x.reshape(-1, hidden_size).contiguous()
        output_size = layer.output_size_per_partition
        acc = torch.zeros(
            flat_x.shape[0],
            output_size,
            device=flat_x.device,
            dtype=torch.float32,
        )

        num_groups = hidden_size // group_size
        weight_q_grouped = layer.weight_q.view(output_size, num_groups, group_size)
        weight_scale_grouped = _group_weight_scale(
            delta_base=layer.weight_delta_base,
            e=layer.weight_e,
            residual_bits=self.quant_config.residual_bits,
        )
        for group_idx in range(num_groups):
            start = group_idx * group_size
            end = start + group_size

            x_group = flat_x[:, start:end].contiguous()
            x_q_group, x_scale_group = torch_npu.npu_dynamic_quant(x_group)

            w_q_group = weight_q_grouped[:, group_idx, :].transpose(0, 1).contiguous()
            # w_scale_group = weight_scale_grouped[:, group_idx].contiguous()

            out_group = torch_npu.npu_quant_matmul(
                x_q_group,
                w_q_group,
                weight_scale_grouped[:, group_idx].contiguous(),
                pertoken_scale=x_scale_group,
                bias=None,
                output_dtype=x.dtype,
            )
            acc.add_(out_group.to(torch.float32))

        out = acc.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*input_shape[:-1], output_size)

    def _apply_quant_group_matmul_fused(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu
        from custom_op.quant_group_matmul.python_api import (
            quant_matmul_k_grouped,
            quant_matmul_k_grouped_dynamic,
        )

        input_shape = x.shape
        hidden_size = input_shape[-1]
        group_size = self.quant_config.residual_group
        flat_x = x.reshape(-1, hidden_size).contiguous()
        output_size = layer.output_size_per_partition
        layer_name = getattr(layer, "prefix", layer.__class__.__name__)

        _maybe_log_nonfinite(f"{layer_name}.input", flat_x)

        num_groups = hidden_size // group_size
        profile = os.getenv("GROUP_QUANT_PROFILE", "0") == "1"
        weight_scale_grouped = getattr(layer, "group_quant_weight_scale_fused", None)
        if weight_scale_grouped is None:
            weight_scale_grouped = _group_weight_scale(
                delta_base=layer.weight_delta_base,
                e=layer.weight_e,
                residual_bits=self.quant_config.residual_bits,
            ).transpose(0, 1).contiguous().to(torch.float32)
        start = _profile_sync_time("start") if profile else None
        if profile:
            start = _profile_sync_time("weight_scale", start)
        _maybe_log_nonfinite(f"{layer_name}.weight_scale_grouped", weight_scale_grouped)
        weight_nz = getattr(layer, "group_quant_weight_nz_fused", None)
        if weight_nz is None:
            weight_nz = _pack_grouped_weight_to_nz(layer.weight_q, group_size)
        if profile:
            start = _profile_sync_time("weight_pack", start)
        block_dim = _get_env_int("GROUP_QUANT_BLOCK_DIM", 8)
        parall_num = _get_env_int("GROUP_QUANT_PARALL_NUM", 4)
        if os.getenv("GROUP_QUANT_USE_DYNAMIC_MATMUL_WRAPPER", "1") == "1":
            out = quant_matmul_k_grouped_dynamic(
                x=flat_x,
                weight_nz_int8=weight_nz,
                scale_f32=weight_scale_grouped,
                group_size=group_size,
                block_dim=block_dim,
                parall_num=parall_num,
            )
        else:
            x_q_packed, x_scale_grouped = _dynamic_quant_by_group(flat_x, group_size)
            _maybe_log_nonfinite(f"{layer_name}.x_scale_grouped", x_scale_grouped)
            out = quant_matmul_k_grouped(
                x_int8=x_q_packed,
                weight_nz_int8=weight_nz,
                scale_f32=weight_scale_grouped,
                pertoken_scale_f32=x_scale_grouped.reshape(-1).contiguous().to(torch.float32),
                k_group_size=group_size,
                k_group_num=num_groups,
                block_dim=block_dim,
                parall_num=parall_num,
            )
        if profile:
            _profile_sync_time("quant_matmul_grouped", start)
            _maybe_log_profile()
        _maybe_log_nonfinite(f"{layer_name}.quant_matmul_out_fp16", out)
        out = out.to(x.dtype).reshape(*input_shape[:-1], output_size)
        _maybe_log_nonfinite(f"{layer_name}.quant_matmul_out_cast", out)
        return out

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "group_quant_has_int8_kernel", False):
            return self._apply_int8_kernel(layer, x, bias)

        x = self.scheme.prepare_activation(layer, x)
        return self._apply_fp16_fallback(layer, x, bias)
