from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _prioritize_group_quant_custom_opp() -> None:
    """Keep this project's custom operators ahead of vLLM-Ascend's OPP.

    vLLM-Ascend prepends its bundled vendor directory from
    ``AscendPlatform.import_kernels``.  The 0.19 structured-MoE worker imports
    this quantization config afterwards, so restore our vendor directory to
    the front before the first ACLNN custom-op lookup.  The GMM2 adapter also
    resolves its op-api library by absolute path because some releases load a
    same-SONAME library even earlier than this hook.
    """
    vendor_root = (
        Path(__file__).resolve().parents[1]
        / "custom_op/build_root/install/vendors/custom_transformer"
    )
    if not vendor_root.is_dir():
        return

    vendor = str(vendor_root)
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    entries = [entry for entry in current.split(":") if entry and entry != vendor]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([vendor, *entries])


_prioritize_group_quant_custom_opp()

from Smoothquant.custom_linear_quant import GroupQuantLinearMethod
from Smoothquant.custom_moe_quant import GroupQuantMoEMethod
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod


class GroupQuantFusedMoEAdapter(FusedMoEMethodBase):
    def __init__(
        self,
        quant_config: "GroupQuantConfig",
        prefix: str,
        layer: torch.nn.Module,
    ) -> None:
        super().__init__(layer.moe_config)
        del prefix
        self.quant_method = GroupQuantMoEMethod(quant_config, layer.moe_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts,
            intermediate_size_per_partition,
            hidden_size,
            params_dtype,
        )
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts,
            intermediate_size_per_partition,
            hidden_size,
            params_dtype,
        )
        per_group_fields = ("weight_delta_base", "weight_e")
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(field in param_key for field in per_group_fields):
                setattr(param, "quant_method", FusedMoeWeightScaleSupported.GROUP.value)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, router_logits, *args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        del layer
        return None

    @property
    def supports_eplb(self) -> bool:
        return bool(getattr(self.quant_method, "supports_eplb", False))


@register_quantization_config("group_quant")
class GroupQuantConfig(QuantizationConfig):
    """vLLM quantization config for SmoothQuant GroupQuant fake quantization."""

    def __init__(
        self,
        weight_bits: int = 8,
        act_bits: int = 8,
        residual_bits: int = 4,
        residual_group: int = 32,
        quantize_weights: bool = True,
        quantize_activations: bool = True,
        quantize_moe: bool = False,
        quantize_linear: bool = True,
    ):
        super().__init__()
        self.weight_bits = int(weight_bits)
        self.act_bits = int(act_bits)
        self.residual_bits = int(residual_bits)
        self.residual_group = int(residual_group)
        self.quantize_weights = bool(quantize_weights)
        self.quantize_activations = bool(quantize_activations)
        self.quantize_moe = bool(quantize_moe)
        self.quantize_linear = bool(quantize_linear)
        self._validate()
        # vllm-ascend expects quant_config.quant_description to exist even for
        # custom quant backends. We keep it lightweight here and let our own
        # methods drive the actual quantization behavior.
        self.quant_description: dict[str, Any] = {
            "quant_method": self.get_name(),
            "weight_bits": self.weight_bits,
            "act_bits": self.act_bits,
            "residual_bits": self.residual_bits,
            "residual_group": self.residual_group,
            "quantize_weights": self.quantize_weights,
            "quantize_activations": self.quantize_activations,
            "quantize_moe": self.quantize_moe,
            "quantize_linear": self.quantize_linear,
        }

    def _validate(self) -> None:
        if not 2 <= self.weight_bits <= 8:
            raise ValueError(
                f"weight_bits must be in [2, 8], but got {self.weight_bits}."
            )
        if not 2 <= self.act_bits <= 8:
            raise ValueError(
                f"act_bits must be in [2, 8], but got {self.act_bits}."
            )
        if not 1 <= self.residual_bits <= 8:
            raise ValueError(
                "residual_bits must be in [1, 8] because residual exponents "
                f"are stored as int8, but got {self.residual_bits}."
            )
        if self.residual_group <= 0:
            raise ValueError(
                f"residual_group must be positive, but got {self.residual_group}."
            )
        if self.quantize_moe and not self.quantize_weights:
            raise ValueError("GroupQuant MoE requires quantize_weights=True.")
        if self.quantize_moe and not self.quantize_activations:
            raise ValueError(
                "GroupQuant MoE is a W8A8 path and requires "
                "quantize_activations=True."
            )
        if self.quantize_moe and self.act_bits != 8:
            raise ValueError(
                "GroupQuant MoE custom kernels currently support act_bits=8 "
                f"only, but got {self.act_bits}."
            )

    def get_name(self) -> str:
        return "group_quant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GroupQuantConfig":
        return cls(
            weight_bits=cls.get_from_keys_or(config, ["weight_bits", "w_bits"], 8),
            act_bits=cls.get_from_keys_or(config, ["act_bits", "a_bits"], 8),
            residual_bits=cls.get_from_keys_or(config, ["residual_bits"], 4),
            residual_group=cls.get_from_keys_or(
                config,
                ["residual_group", "group_size"],
                32,
            ),
            quantize_weights=cls.get_from_keys_or(
                config,
                ["quantize_weights", "weight_quant"],
                True,
            ),
            quantize_activations=cls.get_from_keys_or(
                config,
                ["quantize_activations", "act_quant", "activation_quant"],
                True,
            ),
            quantize_moe=cls.get_from_keys_or(
                config,
                ["quantize_moe", "moe_quant"],
                False,
            ),
            quantize_linear=cls.get_from_keys_or(
                config,
                ["quantize_linear", "linear_quant"],
                True,
            ),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if self.quantize_linear:
                return GroupQuantLinearMethod(self)
            return UnquantizedLinearMethod()
        if isinstance(layer, FusedMoE):
            if self.quantize_moe:
                return GroupQuantFusedMoEAdapter(self, prefix, layer)
            return AscendUnquantizedFusedMoEMethod(layer.moe_config)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []
