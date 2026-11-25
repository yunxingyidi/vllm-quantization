from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
import torch_npu

from Smoothquant.fuse_moe.moe_mlp import (
    _expert_count_from_group_quant_metadata,
    group_quant_apply_mlp,
)
from Smoothquant.group_quant_linear_scheme import (
    dequantize_weight_last_dim_runtime,
    quantize_weight_last_dim_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correctness and latency check for the GroupQuant MoE two-stage path."
    )
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--group-list-type", type=int, choices=[0, 1], default=0)
    parser.add_argument("--zero-last-expert", action="store_true")
    parser.add_argument("--with-topk-scales", action="store_true")
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser


def check_metadata_expert_count(experts: int) -> None:
    """Guard the tensor/list metadata layouts used by profile and EPLB."""
    delta_base = torch.empty(experts, 4)
    exponent = torch.empty(experts, 4, 2)
    assert _expert_count_from_group_quant_metadata(
        delta_base, exponent, name="GMM2"
    ) == experts
    assert _expert_count_from_group_quant_metadata(
        list(delta_base.unbind(0)), list(exponent.unbind(0)), name="GMM2"
    ) == experts


def build_group_list(m: int, experts: int, group_list_type: int, zero_last_expert: bool) -> torch.Tensor:
    counts = torch.zeros(experts, dtype=torch.int64)
    active_experts = experts - 1 if zero_last_expert and experts > 1 else experts
    for token_idx in range(m):
        counts[token_idx % active_experts] += 1
    if group_list_type == 1:
        return counts
    return counts.cumsum(dim=0)


def to_cumsum_group_list(group_list: torch.Tensor, group_list_type: int) -> torch.Tensor:
    if group_list_type == 0:
        return group_list
    return group_list.cumsum(dim=0)


def make_inputs(args: argparse.Namespace):
    if args.k % args.group_size != 0:
        raise ValueError(f"k={args.k} must be divisible by group_size={args.group_size}.")
    torch.manual_seed(args.seed)
    hidden = torch.randn(args.m, args.k, dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(args.experts, 2 * args.n, args.k, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(args.experts, args.k, args.n, dtype=torch.bfloat16) * 0.02
    w1_q, w1_delta, w1_e = quantize_weight_last_dim_runtime(
        w1,
        n_bits=8,
        residual_bits=args.residual_bits,
        residual_group=args.group_size,
    )
    w2_q, w2_delta, w2_e = quantize_weight_last_dim_runtime(
        w2,
        n_bits=8,
        residual_bits=args.residual_bits,
        residual_group=args.group_size,
    )
    group_list = build_group_list(
        args.m,
        args.experts,
        args.group_list_type,
        args.zero_last_expert,
    )
    return hidden, w1_q, w1_delta, w1_e, w2_q, w2_delta, w2_e, group_list


def reference(
    hidden: torch.Tensor,
    w1_q: torch.Tensor,
    w1_delta: torch.Tensor,
    w1_e: torch.Tensor,
    w2_q: torch.Tensor,
    w2_delta: torch.Tensor,
    w2_e: torch.Tensor,
    group_list: torch.Tensor,
    residual_bits: int,
    group_list_type: int,
    topk_scales: torch.Tensor | None,
    swiglu_limit: float,
) -> torch.Tensor:
    w1 = dequantize_weight_last_dim_runtime(w1_q, w1_delta, w1_e, residual_bits, torch.float32)
    w2 = dequantize_weight_last_dim_runtime(w2_q, w2_delta, w2_e, residual_bits, torch.float32)
    out = torch.empty(hidden.shape[0], w2.shape[1], dtype=torch.float32)
    start = 0
    for expert_idx, end_value in enumerate(to_cumsum_group_list(group_list, group_list_type).tolist()):
        end = int(end_value)
        if end <= start:
            start = end
            continue
        gate_up = hidden[start:end].float() @ w1[expert_idx].T
        gate, up = gate_up.chunk(2, dim=-1)
        if swiglu_limit > 1e-8:
            gate = torch.clamp(gate, max=swiglu_limit)
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        act = F.silu(gate) * up
        if topk_scales is not None:
            act = act * topk_scales[start:end].reshape(-1, 1).to(act.dtype)
        out[start:end] = act @ w2[expert_idx].T
        start = end
    return out


def run_op(inputs, args: argparse.Namespace) -> torch.Tensor:
    hidden, w1_q, w1_delta, w1_e, w2_q, w2_delta, w2_e, group_list = inputs
    topk_scales = None
    if args.with_topk_scales:
        topk_scales = torch.linspace(0.5, 1.5, hidden.shape[0], dtype=torch.float32).npu()
    out, _ = group_quant_apply_mlp(
        hidden.npu(),
        w1_q.npu(),
        w2_q.npu(),
        group_list.npu(),
        w1_delta.npu(),
        w1_e.npu(),
        w2_delta.npu(),
        w2_e.npu(),
        args.residual_bits,
        args.group_size,
        group_list_type=args.group_list_type,
        topk_scales=topk_scales,
        fusion=True,
        use_bf16=True,
        swiglu_limit=args.swiglu_limit,
    )
    return out


def main() -> None:
    args = build_parser().parse_args()
    check_metadata_expert_count(args.experts)
    torch_npu.npu.config.allow_internal_format = True
    inputs = make_inputs(args)
    print("config:")
    print(f"  M={args.m}, K={args.k}, N={args.n}, experts={args.experts}")
    print(f"  group_size={args.group_size}, group_num={args.k // args.group_size}")
    print(f"  group_list_type={args.group_list_type}, group_list={inputs[-1].tolist()}")

    out = run_op(inputs, args)
    torch.npu.synchronize()
    out_cpu = out.float().cpu()
    print(f"output: shape={tuple(out.shape)}, dtype={out.dtype}, finite={bool(torch.isfinite(out_cpu).all())}")

    if not args.skip_correctness:
        topk_scales = None
        if args.with_topk_scales:
            topk_scales = torch.linspace(0.5, 1.5, inputs[0].shape[0], dtype=torch.float32)
        ref = reference(
            *inputs,
            residual_bits=args.residual_bits,
            group_list_type=args.group_list_type,
            topk_scales=topk_scales,
            swiglu_limit=args.swiglu_limit,
        )
        abs_diff = (out_cpu - ref).abs()
        print("correctness:")
        print(f"  mean_abs_diff={float(abs_diff.mean()):.8f}")
        print(f"  max_abs_diff={float(abs_diff.max()):.8f}")
        torch.testing.assert_close(out_cpu, ref, atol=args.atol, rtol=0.0)

    for _ in range(args.warmup):
        run_op(inputs, args)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        run_op(inputs, args)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iters
    print(f"latency_ms={elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
