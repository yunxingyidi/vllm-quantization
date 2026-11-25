from __future__ import annotations

import argparse
import importlib.util
import sys
import sysconfig
import time
from pathlib import Path

import torch
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Smoothquant.custom_linear_quant import (
    _pack_grouped_activation_q,
    _pack_grouped_weight_to_nz,
)


def test_full_k_activation_pack_is_identity_layout():
    x_q = torch.arange(3 * 64, dtype=torch.int16).to(torch.int8).reshape(3, 64)
    packed = _pack_grouped_activation_q(x_q, 64)

    assert packed.shape == x_q.shape
    assert packed.is_contiguous()
    assert torch.equal(packed, x_q)


def load_torch_adapter() -> None:
    op_dir = REPO_ROOT / "custom_op/quant_matmul_expert_k_grouped"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    candidates = [
        candidate
        for candidate in sorted(
            (op_dir / "build").glob(
                "lib.*/quant_matmul_expert_k_grouped_torch*.so"
            )
        )
        if extension_suffix and candidate.name.endswith(extension_suffix)
    ]
    if not candidates:
        raise RuntimeError(
            "Torch adapter for the current Python ABI is not built. "
            "Run: python setup.py build_ext"
        )
    spec = importlib.util.spec_from_file_location("quant_matmul_expert_k_grouped_torch", candidates[-1])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def build_inputs(m: int, k: int, n: int, experts: int, group_size: int, seed: int):
    if k % group_size != 0:
        raise ValueError("K must be divisible by group_size")
    if group_size % 16 != 0 or n % 32 != 0:
        raise ValueError("group_size must align to 16 and N must align to 32")
    torch.manual_seed(seed)
    x = (torch.randn(m, k, dtype=torch.bfloat16) * 0.1).npu()
    weight_q = torch.randint(-127, 128, (experts, n, k), dtype=torch.int8).npu()
    group_list = torch.zeros(experts, dtype=torch.int64)
    for token_idx in range(m):
        group_list[token_idx % max(1, experts - 1)] += 1
    if experts == 1:
        group_list[0] = m
    group_list = group_list.npu()
    k_group_num = k // group_size
    x_q_flat, x_scale = torch_npu.npu_dynamic_quant(x)
    x_q = _pack_grouped_activation_q(x_q_flat, group_size)
    weight_scale = (torch.rand(experts, k_group_num, n, dtype=torch.float32) * 0.02 + 1e-4).npu()
    weight_nz = torch.cat([
        _pack_grouped_weight_to_nz(weight_q[e], group_size)
        for e in range(experts)
    ]).contiguous()
    return x_q, weight_q, weight_nz, group_list, weight_scale, x_scale


def run_fallback(x_q, weight_q, group_list, weight_scale, x_scale, group_size):
    from custom_op.quant_group_matmul.python_api import quant_matmul_k_grouped

    k_group_num = weight_scale.shape[1]
    m = x_q.shape[0] // k_group_num
    outputs = []
    start = 0
    for expert_idx, expert_m in enumerate(group_list.cpu().tolist()):
        end = start + int(expert_m)
        if end > start:
            expert_x = x_q.view(k_group_num, m, group_size)[:, start:end].contiguous().view(-1, group_size)
            expert_token_scale = x_scale.reshape(-1)[start:end]
            expert_x_scale = (
                expert_token_scale.reshape(1, -1)
                .expand(k_group_num, -1)
                .contiguous()
                .reshape(-1)
            )
            expert_weight = _pack_grouped_weight_to_nz(weight_q[expert_idx], group_size)
            outputs.append(quant_matmul_k_grouped(
                expert_x,
                expert_weight,
                weight_scale[expert_idx],
                expert_x_scale,
                group_size,
                k_group_num,
            ))
        start = end
    if not outputs:
        return torch.empty((0, weight_scale.shape[-1]), device=x_q.device, dtype=torch.bfloat16)
    return torch.cat(outputs, dim=0).to(torch.bfloat16)


def run_custom(x_q, weight_nz, group_list, weight_scale, x_scale):
    m = x_q.shape[0] if x_q.dim() == 3 else x_q.shape[0] // weight_scale.shape[1]
    if x_scale.numel() != m:
        raise ValueError(f"x_scale must be native per-token [M={m}].")
    return torch.ops._C_ascend.quant_matmul_expert_k_grouped(
        x_q.contiguous(),
        weight_nz.contiguous(),
        group_list.contiguous(),
        weight_scale.reshape(-1, weight_scale.shape[-1]).contiguous(),
        x_scale.reshape(-1).contiguous(),
    )


def benchmark(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--perf-only", action="store_true")
    args = parser.parse_args()

    load_torch_adapter()
    torch_npu.npu.config.allow_internal_format = True
    inputs = build_inputs(args.m, args.k, args.n, args.experts, args.group_size, args.seed)
    x_q, weight_q, weight_nz, group_list, weight_scale, x_scale = inputs
    fallback = None
    precision_result = "skipped (perf-only)"
    if not args.perf_only:
        fallback = run_fallback(x_q, weight_q, group_list, weight_scale, x_scale, args.group_size)
        custom = run_custom(x_q, weight_nz, group_list, weight_scale, x_scale)
        torch.npu.synchronize()
        if custom.dtype != torch.bfloat16:
            raise AssertionError(f"custom GMM2 must return BF16, got {custom.dtype}")
        custom_cpu = custom.cpu()
        fallback_cpu = fallback.cpu()
        torch.testing.assert_close(custom_cpu, fallback_cpu, rtol=2e-2, atol=2e-2)
        precision_result = (
            "exact match" if torch.equal(custom_cpu, fallback_cpu)
            else "within rtol=2e-2, atol=2e-2"
        )

    custom_ms = benchmark(
        lambda: run_custom(x_q, weight_nz, group_list, weight_scale, x_scale),
        args.warmup,
        args.iters,
    )
    print(f"shape: M={args.m}, K={args.k}, N={args.n}, E={args.experts}, group_size={args.group_size}")
    print(f"group_list={group_list.cpu().tolist()}")
    print(f"precision: {precision_result}")
    print(f"custom_ms={custom_ms:.6f}")
    if fallback is not None:
        fallback_ms = benchmark(
            lambda: run_fallback(x_q, weight_q, group_list, weight_scale, x_scale, args.group_size),
            args.warmup,
            args.iters,
        )
        print(f"fallback_ms={fallback_ms:.6f}")
        print(f"speedup={fallback_ms / custom_ms:.3f}x")


if __name__ == "__main__":
    main()
