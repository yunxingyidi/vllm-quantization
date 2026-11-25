from __future__ import annotations

import argparse
import time

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Python grouped dynamic quant with the fused prototype."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser


def get_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def python_grouped_dynamic_quant(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    m, k = x.shape
    if k % group_size != 0:
        raise ValueError(f"k={k} must be divisible by group_size={group_size}")
    group_num = k // group_size
    x_grouped = (
        x.view(m, group_num, group_size)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(group_num * m, group_size)
    )
    x_q, x_scale = torch_npu.npu_dynamic_quant(x_grouped)
    return x_q, x_scale.view(group_num, m).contiguous()


def fused_grouped_dynamic_quant(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from custom_op.quant_group_matmul.python_api import dynamic_quant_k_grouped

    return dynamic_quant_k_grouped(x, group_size)


def benchmark(fn, x: torch.Tensor, group_size: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(x, group_size)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(x, group_size)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> None:
    args = build_parser().parse_args()
    dtype = get_dtype(args.dtype)
    if args.k % args.group_size != 0:
        raise ValueError(f"k={args.k} must be divisible by group_size={args.group_size}")

    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device=args.device, dtype=dtype)

    ref_q, ref_scale = python_grouped_dynamic_quant(x, args.group_size)
    test_q, test_scale = fused_grouped_dynamic_quant(x, args.group_size)

    q_diff = (ref_q.to(torch.int16) - test_q.to(torch.int16)).abs()
    scale_diff = (ref_scale - test_scale).abs()
    print("input config:")
    print(f"  device={args.device}")
    print(f"  m={args.m}, k={args.k}, group_size={args.group_size}")
    print(f"  dtype={dtype}, seed={args.seed}")
    print("correctness:")
    print(f"  q shape={tuple(test_q.shape)}, dtype={test_q.dtype}")
    print(f"  scale shape={tuple(test_scale.shape)}, dtype={test_scale.dtype}")
    print(f"  q exact equal={bool((ref_q == test_q).all().cpu())}")
    print(f"  q max abs diff={int(q_diff.max().cpu())}")
    print(f"  scale max abs diff={float(scale_diff.max().cpu()):.10f}")
    print(f"  scale mean abs diff={float(scale_diff.mean().cpu()):.10f}")

    python_time = benchmark(
        python_grouped_dynamic_quant,
        x,
        args.group_size,
        args.warmup,
        args.iters,
    )
    fused_time = benchmark(
        fused_grouped_dynamic_quant,
        x,
        args.group_size,
        args.warmup,
        args.iters,
    )
    print("timing:")
    print(f"  python grouped dynamic quant: {python_time * 1000:.4f} ms")
    print(f"  fused prototype dynamic quant: {fused_time * 1000:.4f} ms")
    print(f"  speedup: {python_time / fused_time:.3f}x")


if __name__ == "__main__":
    main()
