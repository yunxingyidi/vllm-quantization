from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Optional

import torch
from vllm.forward_context import get_forward_context

from Smoothquant.fuse_moe.moe_mlp import group_quant_apply_mlp, unified_apply_mlp
from Smoothquant.fuse_moe import dispatch_quant_scatter
from vllm_ascend.ops.fused_moe import moe_comm_method as upstream_moe_comm_method

_GROUP_QUANT_FUSED_EXPERTS_PATCHED = False


def _build_fused_experts_result(**kwargs):
    """Build the result across the v0.19 and current structured ABIs."""
    parameters = inspect.signature(
        upstream_moe_comm_method.FusedExpertsResult
    ).parameters
    return upstream_moe_comm_method.FusedExpertsResult(**{
        name: value for name, value in kwargs.items() if name in parameters
    })


def uses_typed_fused_experts_contract() -> bool:
    """Return whether the loaded vLLM-Ascend uses MoEFusedExpertsInput."""
    parameters = inspect.signature(
        upstream_moe_comm_method.MoECommMethod.fused_experts
    ).parameters
    return "fused_experts_input" in parameters


def get_active_moe_comm_method():
    context = get_forward_context()
    method = getattr(context, "moe_comm_method", None)
    if method is not None:
        return method
    extra_context = getattr(upstream_moe_comm_method, "_EXTRA_CTX", None)
    method = getattr(extra_context, "moe_comm_method", None)
    if method is None:
        raise RuntimeError("Missing vLLM-Ascend MoE communication context.")
    return method


def _validate_group_quant_metadata(
    *,
    dynamic_eplb: bool,
    w1_weight_delta_base,
    w1_weight_e,
    w2_weight_delta_base,
    w2_weight_e,
    group_quant_residual_bits,
    group_quant_group_size,
) -> None:
    del dynamic_eplb
    if any(
        value is None
        for value in (
            w1_weight_delta_base,
            w1_weight_e,
            w2_weight_delta_base,
            w2_weight_e,
            group_quant_residual_bits,
            group_quant_group_size,
        )
    ):
        raise ValueError("GroupQuant MoE path requires grouped weight metadata.")


def _install_legacy_override(original_fused_experts) -> None:
    @wraps(original_fused_experts)
    def group_quant_aware_fused_experts(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor | list[torch.Tensor],
        w2: torch.Tensor | list[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_int8_w8a8: bool = False,
        use_int4_w4a8: bool = False,
        use_int4_w4a16: bool = False,
        global_num_experts: Optional[int] = None,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[list[torch.Tensor] | torch.Tensor] = None,
        w2_scale: Optional[list[torch.Tensor] | torch.Tensor] = None,
        w1_scale_bias: torch.Tensor | None = None,
        w2_scale_bias: torch.Tensor | None = None,
        w1_offset: Optional[torch.Tensor] = None,
        w2_offset: Optional[torch.Tensor] = None,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        need_trans: bool = False,
        dynamic_eplb: bool = False,
        mc2_mask: torch.Tensor | None = None,
        pertoken_scale: Optional[torch.Tensor] = None,
        group_quant: bool = False,
        w1_weight_delta_base: torch.Tensor | list[torch.Tensor] | None = None,
        w1_weight_e: torch.Tensor | list[torch.Tensor] | None = None,
        w2_weight_delta_base: torch.Tensor | list[torch.Tensor] | None = None,
        w2_weight_e: torch.Tensor | list[torch.Tensor] | None = None,
        group_quant_residual_bits: int | None = None,
        group_quant_group_size: int | None = None,
        swiglu_limit: float = 0.0,
    ):
        if not group_quant:
            return original_fused_experts(
                self,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                use_int8_w8a8=use_int8_w8a8,
                use_int4_w4a8=use_int4_w4a8,
                use_int4_w4a16=use_int4_w4a16,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                w1_offset=w1_offset,
                w2_offset=w2_offset,
                shared_experts=shared_experts,
                quantized_x_for_share=quantized_x_for_share,
                dynamic_scale_for_share=dynamic_scale_for_share,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                need_trans=need_trans,
                dynamic_eplb=dynamic_eplb,
                mc2_mask=mc2_mask,
                pertoken_scale=pertoken_scale,
            )

        _validate_group_quant_metadata(
            dynamic_eplb=dynamic_eplb,
            w1_weight_delta_base=w1_weight_delta_base,
            w1_weight_e=w1_weight_e,
            w2_weight_delta_base=w2_weight_delta_base,
            w2_weight_e=w2_weight_e,
            group_quant_residual_bits=group_quant_residual_bits,
            group_quant_group_size=group_quant_group_size,
        )
        results = dispatch_quant_scatter.maybe_dispatch(
            self.token_dispatcher,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            log2phy=log2phy,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=dynamic_eplb,
            pertoken_scale=pertoken_scale,
        )
        if results is None:
            results = self.token_dispatcher.token_dispatch(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                log2phy=log2phy,
                apply_router_weight_on_input=apply_router_weight_on_input,
                dynamic_eplb=dynamic_eplb,
                pertoken_scale=pertoken_scale,
                # GroupQuant GMM1 is W8A8. Let init_routing_v2 perform its
                # supported quant_mode=1 while gathering/sorting BF16 input.
                with_quant=True,
            )
        mlp_output = unified_apply_mlp(
            hidden_states=results["hidden_states"],
            w1=w1,
            w2=w2,
            group_list=results["group_list"],
            dynamic_scale=results.get("dynamic_scale"),
            group_list_type=results["group_list_type"],
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
            topk_scales=results.get("topk_scales"),
            fusion=True,
            dynamic_eplb=dynamic_eplb,
            need_trans=need_trans,
            activation=activation,
            group_quant=True,
            w1_scale=w1_scale,
            w1_weight_delta_base=w1_weight_delta_base,
            w1_weight_e=w1_weight_e,
            w2_weight_delta_base=w2_weight_delta_base,
            w2_weight_e=w2_weight_e,
            group_quant_residual_bits=group_quant_residual_bits,
            group_quant_group_size=group_quant_group_size,
            # Routed hidden_states is INT8 after fused dispatch quantization;
            # preserve the model residual dtype instead of inferring FP16 from
            # that transport tensor.
            use_bf16=hidden_states.dtype == torch.bfloat16,
            swiglu_limit=swiglu_limit,
        )
        return self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            context_metadata=results.get("context_metadata"),
        )

    upstream_moe_comm_method.MoECommMethod.fused_experts = (
        group_quant_aware_fused_experts
    )
    fused_mc2_cls = getattr(
        upstream_moe_comm_method, "FusedMC2CommImpl", None
    )
    if (fused_mc2_cls is not None and
            "fused_experts" in fused_mc2_cls.__dict__):
        original_fused_mc2 = fused_mc2_cls.fused_experts

        @wraps(original_fused_mc2)
        def group_quant_aware_fused_mc2(self, *args, **kwargs):
            if kwargs.get("group_quant", False):
                # The monolithic dispatch_ffn_combine ABI has no per-K-group
                # scale input.  Reuse the FusedMC2 dispatch/combine objects and
                # run the GroupQuant GMM chain between them.
                return group_quant_aware_fused_experts(
                    self, *args, **kwargs
                )
            return original_fused_mc2(self, *args, **kwargs)

        fused_mc2_cls.fused_experts = group_quant_aware_fused_mc2


def _install_typed_override(original_fused_experts) -> None:
    # These modules are unavailable in older vLLM-Ascend releases, so keep the
    # imports behind the ABI check.
    from Smoothquant.fuse_moe.moe_runtime_args import build_token_dispatch_input
    from Smoothquant.fuse_moe.moe_stage_contracts import MoEFusedExpertsInput

    @wraps(original_fused_experts)
    def group_quant_aware_fused_experts(self, fused_experts_input):
        if not isinstance(fused_experts_input, MoEFusedExpertsInput):
            return original_fused_experts(
                self, fused_experts_input=fused_experts_input
            )
        weights = fused_experts_input.weights
        if not weights.has_group_quant:
            return original_fused_experts(
                self, fused_experts_input=fused_experts_input
            )

        _validate_group_quant_metadata(
            dynamic_eplb=fused_experts_input.dynamic_eplb,
            w1_weight_delta_base=weights.w1_weight_delta_base,
            w1_weight_e=weights.w1_weight_e,
            w2_weight_delta_base=weights.w2_weight_delta_base,
            w2_weight_e=weights.w2_weight_e,
            group_quant_residual_bits=weights.group_quant_residual_bits,
            group_quant_group_size=weights.group_quant_group_size,
        )
        before_dispatch_evt = torch.npu.current_stream().record_event()
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]
        dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        dispatch_output = self.token_dispatcher.token_dispatch(
            token_dispatch_input=dispatch_input
        )
        w1 = weights.w1_weight_q
        w2 = weights.w2_weight_q
        assert w1 is not None and w2 is not None
        mlp_output, before_gmm2_evt = group_quant_apply_mlp(
            hidden_states=dispatch_output.hidden_states,
            w1=w1,
            w2=w2,
            group_list=dispatch_output.group_list,
            group_list_type=dispatch_output.group_list_type,
            dynamic_scale=dispatch_output.dynamic_scale,
            topk_scales=dispatch_output.topk_scales,
            activation=fused_experts_input.activation,
            fusion=True,
            dynamic_eplb=fused_experts_input.dynamic_eplb,
            w1_fixpipe_scale=weights.w1_scale,
            w1_weight_delta_base=weights.w1_weight_delta_base,
            w1_weight_e=weights.w1_weight_e,
            w2_weight_delta_base=weights.w2_weight_delta_base,
            w2_weight_e=weights.w2_weight_e,
            group_quant_residual_bits=weights.group_quant_residual_bits,
            group_quant_group_size=weights.group_quant_group_size,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )
        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=dispatch_output.combine_metadata,
        )
        return _build_fused_experts_result(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=dispatch_output.group_list_type,
            expert_tokens=dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    upstream_moe_comm_method.MoECommMethod.fused_experts = (
        group_quant_aware_fused_experts
    )
    fused_mc2_cls = getattr(
        upstream_moe_comm_method, "FusedMC2CommImpl", None
    )
    if (fused_mc2_cls is not None and
            "fused_experts" in fused_mc2_cls.__dict__):
        original_fused_mc2 = fused_mc2_cls.fused_experts

        @wraps(original_fused_mc2)
        def group_quant_aware_fused_mc2(self, fused_experts_input):
            if (isinstance(fused_experts_input, MoEFusedExpertsInput) and
                    fused_experts_input.weights.has_group_quant):
                # GroupQuant retains MC2 dispatch/combine, while replacing the
                # unsupported monolithic FFN portion with custom GMM1/GMM2.
                return group_quant_aware_fused_experts(
                    self, fused_experts_input
                )
            return original_fused_mc2(
                self, fused_experts_input=fused_experts_input
            )

        fused_mc2_cls.fused_experts = group_quant_aware_fused_mc2


def install_group_quant_fused_experts_override() -> None:
    global _GROUP_QUANT_FUSED_EXPERTS_PATCHED
    if _GROUP_QUANT_FUSED_EXPERTS_PATCHED:
        return
    dispatch_quant_scatter.initialize()
    original_fused_experts = upstream_moe_comm_method.MoECommMethod.fused_experts
    if uses_typed_fused_experts_contract():
        _install_typed_override(original_fused_experts)
    else:
        _install_legacy_override(original_fused_experts)
    _GROUP_QUANT_FUSED_EXPERTS_PATCHED = True
