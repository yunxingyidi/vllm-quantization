from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from Smoothquant.group_quant_linear_scheme import (
    dequantize_weight_last_dim_runtime,
    quantize_weight_last_dim_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate GroupQuant MoE weight params and reference MLP outputs."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=96)
    parser.add_argument("--num-tokens", type=int, default=17)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-topk-scale", action="store_true")
    return parser


def get_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.bfloat16


def build_group_list(num_tokens: int, num_experts: int) -> torch.Tensor:
    base = torch.zeros(num_experts, dtype=torch.int64)
    for token_idx in range(num_tokens):
        base[token_idx % num_experts] += 1
    return base


def swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def run_reference_moe_mlp(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_scales: torch.Tensor | None,
) -> torch.Tensor:
    outputs = torch.empty(
        hidden_states.shape[0],
        w2.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    start = 0
    for expert_idx, count in enumerate(group_list.tolist()):
        count = int(count)
        if count <= 0:
            continue
        end = start + count
        expert_hidden = hidden_states[start:end]
        gate_up = F.linear(expert_hidden, w1[expert_idx])
        activated = swiglu(gate_up)
        if topk_scales is not None:
            activated = activated * topk_scales[start:end].view(-1, 1)
        outputs[start:end] = F.linear(activated, w2[expert_idx])
        start = end
    return outputs


def summarize_tensor_diff(name: str, ref: torch.Tensor, test: torch.Tensor) -> None:
    diff = (ref.to(torch.float32) - test.to(torch.float32)).abs()
    print(f"{name}:")
    print(f"  ref shape={tuple(ref.shape)}, dtype={ref.dtype}")
    print(f"  test shape={tuple(test.shape)}, dtype={test.dtype}")
    print(f"  mean abs diff={float(diff.mean().cpu()):.8f}")
    print(f"  max  abs diff={float(diff.max().cpu()):.8f}")


def main() -> None:
    args = build_parser().parse_args()
    dtype = get_dtype(args.dtype)

    if args.hidden_size % args.group_size != 0:
        raise ValueError(
            f"hidden_size={args.hidden_size} must be divisible by group_size={args.group_size}"
        )
    if args.intermediate_size % args.group_size != 0:
        raise ValueError(
            f"intermediate_size={args.intermediate_size} must be divisible by group_size={args.group_size}"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    hidden_states = torch.randn(args.num_tokens, args.hidden_size, device=device, dtype=dtype)
    group_list = build_group_list(args.num_tokens, args.num_experts).to(device)
    topk_scales = None
    if args.use_topk_scale:
        topk_scales = torch.rand(args.num_tokens, device=device, dtype=dtype) * 0.5 + 0.5

    w1 = torch.randn(
        args.num_experts,
        2 * args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.2
    w2 = torch.randn(
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.2

    w1_q, w1_delta, w1_e = quantize_weight_last_dim_runtime(
        weight=w1,
        n_bits=args.weight_bits,
        residual_bits=args.residual_bits,
        residual_group=args.group_size,
    )
    w2_q, w2_delta, w2_e = quantize_weight_last_dim_runtime(
        weight=w2,
        n_bits=args.weight_bits,
        residual_bits=args.residual_bits,
        residual_group=args.group_size,
    )

    w1_dequant = dequantize_weight_last_dim_runtime(
        weight_q=w1_q,
        delta_base=w1_delta,
        e=w1_e,
        residual_bits=args.residual_bits,
        output_dtype=dtype,
    )
    w2_dequant = dequantize_weight_last_dim_runtime(
        weight_q=w2_q,
        delta_base=w2_delta,
        e=w2_e,
        residual_bits=args.residual_bits,
        output_dtype=dtype,
    )

    print("config:")
    print(f"  device={device}")
    print(f"  dtype={dtype}")
    print(f"  num_experts={args.num_experts}")
    print(f"  hidden_size={args.hidden_size}")
    print(f"  intermediate_size={args.intermediate_size}")
    print(f"  num_tokens={args.num_tokens}")
    print(f"  group_size={args.group_size}")
    print(f"  residual_bits={args.residual_bits}")
    print(f"  group_list={group_list.cpu().tolist()}")
    print("parameter shapes:")
    print(f"  w1_q={tuple(w1_q.shape)}, w1_delta={tuple(w1_delta.shape)}, w1_e={tuple(w1_e.shape)}")
    print(f"  w2_q={tuple(w2_q.shape)}, w2_delta={tuple(w2_delta.shape)}, w2_e={tuple(w2_e.shape)}")

    summarize_tensor_diff("w1 dequant vs original", w1, w1_dequant)
    summarize_tensor_diff("w2 dequant vs original", w2, w2_dequant)

    ref_out = run_reference_moe_mlp(
        hidden_states=hidden_states,
        group_list=group_list,
        w1=w1,
        w2=w2,
        topk_scales=topk_scales,
    )
    quant_out = run_reference_moe_mlp(
        hidden_states=hidden_states,
        group_list=group_list,
        w1=w1_dequant,
        w2=w2_dequant,
        topk_scales=topk_scales,
    )
    summarize_tensor_diff("moe mlp output", ref_out, quant_out)


if __name__ == "__main__":
    main()
