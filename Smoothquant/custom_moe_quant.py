import os
import math
from collections.abc import Callable
from dataclasses import replace
from inspect import signature
from typing import Any

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.flash_common3_context import get_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts

from Smoothquant.fuse_moe.moe_comm_method import (
    get_active_moe_comm_method,
    install_group_quant_fused_experts_override,
    uses_typed_fused_experts_contract,
)
from Smoothquant.fuse_moe.moe_mlp import (
    _pack_expert_gmm2_scale,
    _use_bf16_expert_gmm2_scale,
)
from Smoothquant.custom_linear_quant import (
    _group_weight_scale,
    _pack_grouped_weight_to_nz,
)
from Smoothquant.group_quant_linear_scheme import quantize_weight_last_dim_runtime
from .group_quant_moe_scheme import AscendMoEScheme, QuantType, get_moe_num_logical_experts


_GROUP_QUANT_EPLB_WEIGHT_NAMES = [
    "w13_weight_q_list",
    "w13_weight_delta_base_list",
    "w13_weight_e_list",
    "w2_weight_q_list",
    "w2_weight_delta_base_list",
    "w2_weight_e_list",
]
_GROUP_QUANT_EPLB_ADAPTOR_PATCHED = False
_SELECT_EXPERT_PARAMETERS = signature(select_experts).parameters


def _resolve_moe_group_size(last_dim: int, requested_group_size: int) -> int:
    """Choose a TP-local group size compatible with a projection's K dim.

    Qwen3-30B-A3B has K=2048 for GMM1 but TP-local K=384 for GMM2.
    A single requested size of 256 therefore maps to 256 and 128
    respectively. Restricting the fallback to a divisor of the requested
    group keeps the effective quantization granularity no coarser than asked.
    """
    group_size = math.gcd(int(last_dim), int(requested_group_size))
    if group_size < 16 or group_size % 16 != 0:
        raise ValueError(
            "No GroupQuant group size aligned to 16 is compatible with "
            f"last_dim={last_dim} and requested_group_size={requested_group_size}."
        )
    return group_size


def _resolve_gmm2_group_size(last_dim: int, requested_group_size: int) -> int:
    """Resolve GMM2 independently so its TP-local K can use a full-K group.

    A value of 0 means the complete TP-local K dimension; otherwise the
    configured value must divide K.  The default in env_vllm_ascend.sh is 0,
    validated on the Qwen3-30B-A3B model-level PPL benchmark.
    """
    configured = os.getenv("GROUP_QUANT_GMM2_GROUP_SIZE")
    if configured is None or configured == "":
        return _resolve_moe_group_size(last_dim, requested_group_size)
    try:
        group_size = int(configured)
    except ValueError as exc:
        raise ValueError(
            "GROUP_QUANT_GMM2_GROUP_SIZE must be an integer."
        ) from exc
    if group_size == 0:
        group_size = int(last_dim)
    if (
        group_size < 16
        or group_size % 16 != 0
        or group_size > last_dim
        or last_dim % group_size != 0
    ):
        raise ValueError(
            "GROUP_QUANT_GMM2_GROUP_SIZE must be 0 (full TP-local K) or an "
            f"aligned divisor of K={last_dim}, but got {configured}."
        )
    return group_size


def _install_group_quant_eplb_adaptor() -> None:
    """Teach the vLLM-Ascend EPLB mover about all GroupQuant tensors.

    Upstream W8A8 registers four per-expert tensors.  GroupQuant has three
    tensors for each projection, and moving only the INT8 payload would pair an
    expert with stale residual metadata after a rebalance.
    """
    global _GROUP_QUANT_EPLB_ADAPTOR_PATCHED
    if _GROUP_QUANT_EPLB_ADAPTOR_PATCHED:
        return
    try:
        from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
    except ImportError:
        # Older vLLM-Ascend releases do not provide Dynamic EPLB.
        return

    original_init = VllmEplbAdaptor.init_expert_param_per_layer

    def group_quant_aware_init(self):
        first_experts = self.model.model.layers[
            self.num_dense_layers
        ].mlp.experts
        names = getattr(
            first_experts, "group_quant_eplb_weight_names", None
        )
        if names is None:
            return original_init(self)

        self.param_dict = {}
        self.expert_weight_names = list(names)
        for layer_idx in range(
            self.num_dense_layers, self.config.num_hidden_layers
        ):
            self.expert_param_per_layer[layer_idx] = []
            experts = self.model.model.layers[layer_idx].mlp.experts
            for name in self.expert_weight_names:
                param_key = f"model.layers.{layer_idx}.mlp.experts.{name}"
                self.param_dict[param_key] = getattr(experts, name)
            for local_expert_id in range(self.num_local_experts):
                self.expert_param_per_layer[layer_idx].append([
                    self.param_dict[
                        f"model.layers.{layer_idx}.mlp.experts.{name}"
                    ][local_expert_id]
                    for name in self.expert_weight_names
                ])

    VllmEplbAdaptor.init_expert_param_per_layer = group_quant_aware_init
    _GROUP_QUANT_EPLB_ADAPTOR_PATCHED = True

try:
    from .registry import register_scheme
except ImportError:
    def register_scheme(*_args, **_kwargs):
        def decorator(cls):
            return cls
        return decorator


def zero_experts_compute(expert_indices, expert_scales, num_experts,
                         zero_expert_type, hidden_states):
    if zero_expert_type != "identity":
        raise NotImplementedError(f"Unsupported zero expert type: {zero_expert_type}")
    normal_mask = expert_indices < num_experts
    zero_scales = torch.where(normal_mask, 0.0, expert_scales.clone())
    result = (hidden_states.unsqueeze(1) * zero_scales.unsqueeze(2)).sum(dim=1)
    zero_mask = expert_indices >= num_experts
    return (torch.where(zero_mask, 0, expert_indices),
            torch.where(zero_mask, 0.0, expert_scales), result)


@register_scheme("W8A8_DYNAMIC", "moe")
class GroupQuantFusedMoEMethod(AscendMoEScheme):
    quant_type: QuantType = QuantType.W8A8

    def __init__(self, quant_config=None, moe=None, *, smoothquant_scale=None):
        self.quant_config = quant_config
        self.moe = moe
        self.smoothquant_scale = smoothquant_scale
        self.ep_group = get_ep_group()
        install_group_quant_fused_experts_override()
        _install_group_quant_eplb_adaptor()
        config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not config.model_config.enforce_eager
        )
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        eplb_config = getattr(ascend_config, "eplb_config", None)
        self.dynamic_eplb = bool(
            getattr(eplb_config, "dynamic_eplb", False)
            if eplb_config is not None
            else getattr(ascend_config, "dynamic_eplb", False)
        )
        self.in_dtype = config.model_config.dtype
        self.supports_eplb = True
        # TokenDispatcher owns HCCL setup; a second MC2 communicator here
        # causes duplicate port binding during ordinary TP/EP startup.
        self.moe_all_to_all_group_name = ""

    def get_weight(self, num_experts, intermediate_size_per_partition,
                   hidden_sizes, params_dtype):
        return {
            "w13_weight": torch.empty(num_experts,
                2 * intermediate_size_per_partition, hidden_sizes,
                dtype=params_dtype),
            "w2_weight": torch.empty(num_experts, hidden_sizes,
                intermediate_size_per_partition, dtype=params_dtype),
        }

    def get_dynamic_quant_param(self, num_experts,
                                intermediate_size_per_partition,
                                hidden_sizes, params_dtype):
        del params_dtype
        requested_rg = self.quant_config.residual_group
        w13_rg = _resolve_moe_group_size(hidden_sizes, requested_rg)
        w2_rg = _resolve_gmm2_group_size(
            intermediate_size_per_partition, requested_rg
        )
        return {
            "w13_weight_q": torch.empty(num_experts,
                2 * intermediate_size_per_partition, hidden_sizes,
                dtype=torch.int8),
            "w13_weight_delta_base": torch.empty(num_experts,
                2 * intermediate_size_per_partition, dtype=torch.bfloat16),
            "w13_weight_e": torch.empty(num_experts,
                2 * intermediate_size_per_partition, hidden_sizes // w13_rg,
                dtype=torch.int8),
            "w2_weight_q": torch.empty(num_experts, hidden_sizes,
                intermediate_size_per_partition, dtype=torch.int8),
            "w2_weight_delta_base": torch.empty(num_experts, hidden_sizes,
                dtype=torch.bfloat16),
            "w2_weight_e": torch.empty(num_experts, hidden_sizes,
                intermediate_size_per_partition // w2_rg, dtype=torch.int8),
        }

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              router_logits: torch.Tensor, use_grouped_topk: bool, top_k: int,
              renormalize: bool, topk_group: int | None = None,
              num_expert_group: int | None = None,
              custom_routing_function: Callable | None = None,
              scoring_func: str = "softmax", routed_scaling_factor: float = 1.0,
              e_score_correction_bias: torch.Tensor | None = None,
              global_num_experts: int = -1,
              num_experts: int | None = None,
              expert_map: torch.Tensor | None = None,
              apply_router_weight_on_input: bool = False,
              enable_force_load_balance: bool = False,
              shared_experts: Any | None = None,
              quantized_x_for_share: torch.Tensor | None = None,
              dynamic_scale_for_share: torch.Tensor | None = None,
              pertoken_scale: Any | None = None, activation: str = "silu",
              log2phy: torch.Tensor | None = None,
              global_redundant_expert_num: int = 0,
              mc2_mask: torch.Tensor | None = None,
              tid2eid: torch.Tensor | None = None, **kwargs):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported GroupQuant MoE arguments: {unknown}")
        if num_experts is not None:
            global_num_experts = num_experts
        if global_num_experts <= 0:
            global_num_experts = router_logits.shape[1]
        zero_num = getattr(layer, "zero_expert_num", 0)
        zero_type = getattr(layer, "zero_expert_type", None)
        logical_num = get_moe_num_logical_experts(
            layer, global_num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=getattr(layer, "n_shared_experts", 0) or 0)
        if zero_num == 0 or zero_type is None:
            assert router_logits.shape[1] == logical_num, (
                "[vllm-ascend/W8A8_DYNAMIC] Number of global experts mismatch "
                f"(excluding redundancy): {router_logits.shape[1]} != {logical_num}")
        if self.multistream_overlap_gate:
            context = get_flash_common3_context()
            assert context is not None
            topk_weights, topk_ids = context.topk_weights, context.topk_ids
        else:
            select_kwargs = dict(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
            )
            if "num_logical_experts" in _SELECT_EXPERT_PARAMETERS:
                select_kwargs.update(
                    mix_placement=bool(getattr(layer, "mix_placement", False)),
                    num_logical_experts=router_logits.shape[1],
                    num_shared_experts=getattr(layer, "n_shared_experts", 0) or 0,
                    num_experts=logical_num,
                    tid2eid=tid2eid,
                )
            else:
                select_kwargs["global_num_experts"] = logical_num
            # The structured MoE ABI landed before all routing extensions did:
            # v0.19 accepts num_logical_experts/mix_placement but not tid2eid,
            # while current main accepts the complete set.  Filter individual
            # optional fields instead of treating either signature as a single
            # version bundle.
            select_kwargs = {
                name: value
                for name, value in select_kwargs.items()
                if name in _SELECT_EXPERT_PARAMETERS
            }
            topk_weights, topk_ids = select_experts(**select_kwargs)
        assert topk_ids is not None and topk_weights is not None
        if zero_num > 0 and zero_type is not None:
            topk_ids, topk_weights, zero_result = zero_experts_compute(
                topk_ids, topk_weights, logical_num, zero_type, x)
        if enable_force_load_balance:
            random = torch.rand(topk_ids.size(0), logical_num,
                                device=topk_ids.device)
            topk_ids = torch.argsort(random, dim=1)[:, :topk_ids.size(1)].to(topk_ids.dtype)
        topk_weights = topk_weights.to(self.in_dtype)
        if not getattr(layer, "group_quant_weight_ready", False):
            raise RuntimeError("GroupQuant MoE weights are not ready. Check process_weights_after_loading().")
        inverse_scale = getattr(layer, "group_quant_smoothquant_inverse_scale", None)
        if inverse_scale is not None:
            if pertoken_scale is not None or not x.is_floating_point() or shared_experts is not None:
                raise ValueError("MoE SmoothQuant requires floating-point routed inputs before activation quantization.")
            # Router selection and zero-expert handling above use the original x.
            x = x * inverse_scale
        if self.dynamic_eplb:
            w13_weight_q = layer.w13_weight_q_list
            w13_weight_delta_base = layer.w13_weight_delta_base_list
            w13_weight_e = layer.w13_weight_e_list
            w2_weight_q = layer.w2_weight_q_list
            w2_weight_delta_base = layer.w2_weight_delta_base_list
            w2_weight_e = layer.w2_weight_e_list
            w13_fixpipe_scale = getattr(
                layer, "w13_weight_fixpipe_scale_list", None
            )
        elif hasattr(layer, "w13_weight_q_nz_list"):
            w13_weight_q = layer.w13_weight_q_nz_list
            w13_weight_delta_base = layer.w13_weight_delta_base
            w13_weight_e = layer.w13_weight_e
            w2_weight_q = layer.w2_weight_q
            w2_weight_delta_base = layer.w2_weight_delta_base
            w2_weight_e = layer.w2_weight_e
            w13_fixpipe_scale = getattr(
                layer, "w13_weight_fixpipe_scale", None
            )
        else:
            w13_weight_q = layer.w13_weight_q
            w13_weight_delta_base = layer.w13_weight_delta_base
            w13_weight_e = layer.w13_weight_e
            w2_weight_q = layer.w2_weight_q
            w2_weight_delta_base = layer.w2_weight_delta_base
            w2_weight_e = layer.w2_weight_e
            w13_fixpipe_scale = getattr(layer, "w13_weight_fixpipe_scale", None)
        moe_comm_method = get_active_moe_comm_method()
        if uses_typed_fused_experts_contract():
            from Smoothquant.fuse_moe.moe_runtime_args import (
                MoEFusedExpertsInput,
                MoERoutingParams,
                build_fused_experts_input,
            )

            packed_scale = getattr(layer, "w2_packed_scale", None)
            prepared = getattr(layer, "group_quant_moe_payload", None)
            use_early_combine_prep = getattr(layer, "group_quant_use_early_combine_prep", False)
            # Parameter replacement or module moves can swap tensor objects.
            # Rebuild the regular payload for such calls using current weights.
            if prepared is not None and (
                prepared[0].w1 is w13_weight_q
                and prepared[0].w1_scale is w13_fixpipe_scale
                and prepared[0].w1_weight_delta_base is w13_weight_delta_base
                and prepared[0].w1_weight_e is w13_weight_e
                and prepared[0].w2 is w2_weight_q
                and prepared[0].w2_weight_delta_base is w2_weight_delta_base
                and prepared[0].w2_weight_e is w2_weight_e
                and prepared[0].w2_packed_scale is packed_scale
            ):
                weights, quant = prepared
                fused_experts_input = MoEFusedExpertsInput(
                    hidden_states=x,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    weights=weights,
                    quant=quant,
                    routing=MoERoutingParams(
                        expert_map=expert_map,
                        global_redundant_expert_num=global_redundant_expert_num,
                        mc2_mask=mc2_mask,
                        apply_router_weight_on_input=apply_router_weight_on_input,
                        log2phy=log2phy,
                        pertoken_scale=pertoken_scale,
                    ),
                    activation=activation,
                    swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                    use_early_combine_prep=use_early_combine_prep,
                )
            else:
                fused_experts_input = build_fused_experts_input(
                    hidden_states=x,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    w1=w13_weight_q,
                    w2=w2_weight_q,
                    quant_type=self.quant_type,
                    dynamic_eplb=self.dynamic_eplb,
                    expert_map=expert_map,
                    global_redundant_expert_num=global_redundant_expert_num,
                    mc2_mask=mc2_mask,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    log2phy=log2phy,
                    pertoken_scale=pertoken_scale,
                    activation=activation,
                    need_trans=False,
                    w1_scale=w13_fixpipe_scale,
                    w1_weight_q=w13_weight_q,
                    w1_weight_delta_base=w13_weight_delta_base,
                    w1_weight_e=w13_weight_e,
                    w2_weight_q=w2_weight_q,
                    w2_weight_delta_base=w2_weight_delta_base,
                    w2_weight_e=w2_weight_e,
                    group_quant_residual_bits=self.quant_config.residual_bits,
                    group_quant_group_size=self.quant_config.residual_group,
                    swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                    w2_packed_scale=packed_scale,
                    use_early_combine_prep=use_early_combine_prep,
                )
            output = moe_comm_method.fused_experts(fused_experts_input=fused_experts_input)
        else:
            output = moe_comm_method.fused_experts(
                hidden_states=x, topk_weights=topk_weights, topk_ids=topk_ids,
                w1=w13_weight_q, w2=w2_weight_q,
                activation=activation, expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                shared_experts=shared_experts,
                quantized_x_for_share=quantized_x_for_share,
                dynamic_scale_for_share=dynamic_scale_for_share,
                log2phy=log2phy, global_num_experts=logical_num,
                global_redundant_expert_num=global_redundant_expert_num,
                need_trans=False, dynamic_eplb=self.dynamic_eplb,
                mc2_mask=mc2_mask, pertoken_scale=pertoken_scale,
                w1_scale=w13_fixpipe_scale, w2_scale=None, w1_scale_bias=None,
                w2_scale_bias=None, group_quant=True,
                w1_weight_delta_base=w13_weight_delta_base,
                w1_weight_e=w13_weight_e,
                w2_weight_delta_base=w2_weight_delta_base,
                w2_weight_e=w2_weight_e,
                group_quant_residual_bits=self.quant_config.residual_bits,
                group_quant_group_size=self.quant_config.residual_group,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0))
        if zero_num > 0 and zero_type is not None:
            if hasattr(output, "routed_out"):
                output = replace(output, routed_out=output.routed_out + zero_result)
            else:
                output += zero_result
        return output

    def process_weights_after_loading(self, layer):
        layer.group_quant_weight_ready = False
        if self.quant_config is None or not self.quant_config.quantize_weights:
            raise RuntimeError("GroupQuant MoE main path requires quantize_weights=True.")
        common_args = dict(n_bits=self.quant_config.weight_bits,
                           residual_bits=self.quant_config.residual_bits)
        w13_group_size = _resolve_moe_group_size(
            layer.w13_weight.shape[-1], self.quant_config.residual_group
        )
        w2_group_size = _resolve_gmm2_group_size(
            layer.w2_weight.shape[-1], self.quant_config.residual_group
        )
        w13_source = layer.w13_weight.data
        smoothquant_scale = getattr(self, "smoothquant_scale", None)
        if smoothquant_scale is not None:
            if smoothquant_scale.numel() != w13_source.shape[-1]:
                raise ValueError(
                    f"MoE SmoothQuant expected {w13_source.shape[-1]} input scales, "
                    f"got {smoothquant_scale.numel()}."
                )
            if getattr(layer, "_shared_experts", None) is not None:
                raise ValueError("MoE SmoothQuant currently supports routed-only MoE layers.")
            scale = smoothquant_scale.to(dtype=w13_source.dtype)
            inverse_scale = scale.float().reciprocal().to(self.in_dtype)
            if not bool((torch.isfinite(inverse_scale) & (inverse_scale > 0)).all()):
                raise ValueError("MoE SmoothQuant scales are outside the supported dtype range.")
            w13_source = w13_source * scale.to(device=w13_source.device)
        try:
            w13_q, w13_delta, w13_e = quantize_weight_last_dim_runtime(
                weight=w13_source,
                residual_group=w13_group_size,
                **common_args)
            del w13_source
            w2_q, w2_delta, w2_e = quantize_weight_last_dim_runtime(
                weight=layer.w2_weight.data,
                residual_group=w2_group_size,
                **common_args)
        except ValueError as exc:
            raise RuntimeError(f"GroupQuant MoE weight quantization failed: {exc}") from exc
        for target, value in (
            (layer.w13_weight_q, w13_q),
            (layer.w13_weight_delta_base, w13_delta),
            (layer.w13_weight_e, w13_e),
            (layer.w2_weight_q, w2_q),
            (layer.w2_weight_delta_base, w2_delta),
            (layer.w2_weight_e, w2_e)):
            target.data.copy_(value.contiguous())

        use_vecin_gmm1 = (
            os.getenv("GROUP_QUANT_USE_GMM1_VECIN_FULLK", "0") == "1"
        )
        use_fused_gmm1 = (
            os.getenv("GROUP_QUANT_USE_FUSED_SWIGLU_OP", "0") == "1"
            or use_vecin_gmm1
        )
        use_expert_gmm2 = os.getenv("GROUP_QUANT_USE_EXPERT_GMM2_OP", "0") == "1"
        gmm1_backend = os.getenv("GROUP_QUANT_GMM1_BACKEND", "auto")
        if gmm1_backend not in ("auto", "native", "custom"):
            raise ValueError("GROUP_QUANT_GMM1_BACKEND must be auto, native or custom")
        use_native_gmm1 = (
            gmm1_backend in ("auto", "native")
            and use_fused_gmm1
            and not self.dynamic_eplb
            and w13_group_size == layer.w13_weight_q.shape[-1] == 2048
            and 0 < layer.w13_weight_q.shape[-2] <= 1536
            and layer.w13_weight_q.shape[-2] % 64 == 0
            and getattr(layer, "swiglu_limit", 0.0) <= 1e-8
            and hasattr(torch_npu, "npu_grouped_matmul_swiglu_quant")
        )
        if use_fused_gmm1:
            # Format weights once at load: native GMM1 keeps the complete NZ
            # tensor, while custom GMM1 consumes a tensor per expert. Neither
            # path formats or packs weights during forward.
            w13_weight_q_nz = torch_npu.npu_format_cast(
                layer.w13_weight_q.data.transpose(1, 2).contiguous(), 29
            )
            if use_native_gmm1:
                # Native fused GMM1 consumes one complete [E,K,N] NZ weight.
                # Replace the ND allocation; do not keep a duplicate or pack
                # expert tensors in the forward path.
                layer.w13_weight_q.data = w13_weight_q_nz
            else:
                # Clone removes the expert dimension from the private storage
                # descriptor (5-D -> the 4-D NZ shape required by custom ACLNN).
                layer.w13_weight_q_nz_list = [
                    expert_weight.clone()
                    for expert_weight in w13_weight_q_nz.unbind(dim=0)
                ]
                del layer.w13_weight_q
            del w13_weight_q_nz
        if use_native_gmm1:
            layer.w13_weight_fixpipe_scale = _group_weight_scale(
                w13_delta, w13_e, self.quant_config.residual_bits, use_moe=True,
            ).squeeze(-1).contiguous().to(torch.float32)
        elif use_fused_gmm1 and not use_vecin_gmm1 and not self.dynamic_eplb:
            # The ACLNN GMM1 ABI takes a TensorList, so the per-expert scale
            # layout used by the kernel can be materialized once here.  These
            # tensors are derived solely from immutable quantized weights; the
            # old forward path rebuilt and reformatted them for every layer.
            w13_weight_scale = _group_weight_scale(
                w13_delta,
                w13_e,
                self.quant_config.residual_bits,
                use_moe=True,
            )
            layer.w13_weight_fixpipe_scale = [
                scale.transpose(0, 1).contiguous().to(torch.float32).reshape(-1)
                for scale in w13_weight_scale.unbind(dim=0)
            ]
            del w13_weight_scale
        if use_vecin_gmm1 and not use_native_gmm1:
            # FixPipe accepts per-channel dequant parameters as packed UINT64.
            # Materialize them once while loading weights; doing this in every
            # forward would add another NPU task and erase the local-output
            # kernel's launch saving.
            w13_weight_scale = _group_weight_scale(
                w13_delta,
                w13_e,
                self.quant_config.residual_bits,
                use_moe=True,
            ).transpose(1, 2).contiguous().to(torch.float32)
            # TransQuantParamV3 accepts a flat scale vector on the deployed
            # CANN release, but rejects the native [E, G, N] rank-3 tensor.
            # Packing is elementwise, so flatten once and restore the logical
            # view without changing its byte layout.
            layer.w13_weight_fixpipe_scale = torch_npu.npu_trans_quant_param(
                w13_weight_scale.reshape(-1)
            ).reshape_as(w13_weight_scale).contiguous()
            del w13_weight_scale
        if self.dynamic_eplb:
            if use_fused_gmm1:
                layer.w13_weight_q_list = layer.w13_weight_q_nz_list
                del layer.w13_weight_q_nz_list
            else:
                layer.w13_weight_q_list = [
                    value.clone() for value in layer.w13_weight_q.data.unbind(0)
                ]
                del layer.w13_weight_q
            layer.w13_weight_delta_base_list = [
                value.clone()
                for value in layer.w13_weight_delta_base.data.unbind(0)
            ]
            layer.w13_weight_e_list = [
                value.clone() for value in layer.w13_weight_e.data.unbind(0)
            ]
            if use_vecin_gmm1:
                layer.w13_weight_fixpipe_scale_list = [
                    value.clone()
                    for value in layer.w13_weight_fixpipe_scale.unbind(0)
                ]
                del layer.w13_weight_fixpipe_scale
            if use_expert_gmm2:
                layer.w2_weight_q_list = [
                    _pack_grouped_weight_to_nz(
                        value, w2_group_size
                    ).contiguous()
                    for value in layer.w2_weight_q.data.unbind(0)
                ]
            else:
                layer.w2_weight_q_list = [
                    value.clone() for value in layer.w2_weight_q.data.unbind(0)
                ]
            layer.w2_weight_delta_base_list = [
                value.clone()
                for value in layer.w2_weight_delta_base.data.unbind(0)
            ]
            layer.w2_weight_e_list = [
                value.clone() for value in layer.w2_weight_e.data.unbind(0)
            ]
            for name in (
                "w13_weight_delta_base",
                "w13_weight_e",
                "w2_weight_q",
                "w2_weight_delta_base",
                "w2_weight_e",
            ):
                if name in layer._parameters:
                    delattr(layer, name)
            eplb_weight_names = list(_GROUP_QUANT_EPLB_WEIGHT_NAMES)
            if use_vecin_gmm1:
                eplb_weight_names.append("w13_weight_fixpipe_scale_list")
            layer.group_quant_eplb_weight_names = tuple(eplb_weight_names)
        elif os.getenv("GROUP_QUANT_BENCH_NATIVE_GMM2", "0") == "1":
            if w2_group_size != layer.w2_weight_q.shape[-1]:
                raise ValueError("Native benchmark GMM2 requires one full-K group")
            # In a full-K group the residual exponent is zero. Match
            # _group_weight_scale's BF16 minimum before native GMM consumes it.
            layer.w2_weight_delta_base.data.clamp_(min=1e-5)
            layer.w2_weight_q.data = torch_npu.npu_format_cast(
                layer.w2_weight_q.data.transpose(1, 2).contiguous(), 29
            )
        elif use_expert_gmm2:
            # GMM2 consumes all expert weights as one packed buffer. Replace
            # the ND parameter during post-load so the first forward does not
            # create a second model-sized weight allocation in a global cache.
            packed_w2 = torch.cat(
                [
                    _pack_grouped_weight_to_nz(
                        layer.w2_weight_q.data[expert_idx],
                        w2_group_size,
                    )
                    for expert_idx in range(layer.w2_weight_q.shape[0])
                ]
            ).contiguous()
            layer.w2_weight_q.data = packed_w2
            layer.group_quant_w2_packed = True
            if (
                uses_typed_fused_experts_contract()
                and os.getenv("GROUP_QUANT_GMM2_PREPACK_SCALE", "1") == "1"
            ):
                layer.register_buffer(
                    "w2_packed_scale",
                    _pack_expert_gmm2_scale(
                        layer.w2_weight_delta_base.data,
                        layer.w2_weight_e.data,
                        self.quant_config.residual_bits,
                        use_bf16_scale=_use_bf16_expert_gmm2_scale(layer.w2_weight_e),
                    ),
                    persistent=False,
                )
        if (
            not self.dynamic_eplb
            and uses_typed_fused_experts_contract()
            and os.getenv("GROUP_QUANT_MOE_STATIC_METADATA", "1") == "1"
        ):
            from Smoothquant.fuse_moe.moe_runtime_args import MoEQuantParams, MoEWeights

            w1 = getattr(layer, "w13_weight_q_nz_list", None)
            if w1 is None:
                w1 = layer.w13_weight_q
            weights = MoEWeights(
                w1=w1,
                w2=layer.w2_weight_q,
                w1_scale=getattr(layer, "w13_weight_fixpipe_scale", None),
                w1_weight_q=w1,
                w1_weight_delta_base=layer.w13_weight_delta_base,
                w1_weight_e=layer.w13_weight_e,
                w2_weight_q=layer.w2_weight_q,
                w2_weight_delta_base=layer.w2_weight_delta_base,
                w2_weight_e=layer.w2_weight_e,
                w2_packed_scale=getattr(layer, "w2_packed_scale", None),
                group_quant_residual_bits=self.quant_config.residual_bits,
                group_quant_group_size=self.quant_config.residual_group,
                group_quant_k_group_sizes=(w13_group_size, w2_group_size),
            )
            if not weights.has_group_quant:
                raise ValueError("GroupQuant MoE path requires grouped weight metadata.")
            layer.group_quant_moe_payload = (weights, MoEQuantParams(quant_type=self.quant_type))
        if (
            uses_typed_fused_experts_contract()
            and os.getenv("GROUP_QUANT_MOE_EARLY_COMBINE_PREP", "1") == "1"
        ):
            from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE, AscendSharedFusedMoE

            # Shared experts use combine events to sequence work on another
            # stream. Only the known routed-only layers can opt out.
            layer.group_quant_use_early_combine_prep = (
                type(layer) is AscendFusedMoE
                or (type(layer) is AscendSharedFusedMoE and layer._shared_experts is None)
            )
        # BF16 tensors are checkpoint-loading sources only. All runtime paths,
        # including the fallback, reconstruct from GroupQuant state.
        if "w13_weight" in layer._parameters:
            del layer.w13_weight
        if "w2_weight" in layer._parameters:
            del layer.w2_weight
        if smoothquant_scale is not None:
            layer.register_buffer(
                "group_quant_smoothquant_inverse_scale",
                inverse_scale.to(device=layer.w13_weight_delta_base.device),
                persistent=False,
            )
        layer.group_quant_weight_ready = True


GroupQuantMoEMethod = GroupQuantFusedMoEMethod
