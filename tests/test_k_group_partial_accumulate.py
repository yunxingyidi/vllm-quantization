from __future__ import annotations

import argparse

import torch

from Smoothquant.custom_linear_quant import (
    _dynamic_quant_by_group,
    _group_weight_scale,
    _pack_grouped_weight_to_nz,
)
from custom_op.quant_group_matmul.python_api import quant_matmul_grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Python-loop group quant matmul against K-group device kernel."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--dump-partials", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--isolate-groups", action="store_true")
    parser.add_argument("--check-single-group", action="store_true")
    parser.add_argument("--check-k-axis-single-group", action="store_true")
    parser.add_argument("--block-dim", type=int, default=8)
    return parser


def get_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def build_inputs(
    m: int,
    k: int,
    n: int,
    group_size: int,
    residual_bits: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    if k % group_size != 0:
        raise ValueError(f"k={k} must be divisible by group_size={group_size}")
    if group_size % 16 != 0:
        raise ValueError(f"group_size={group_size} must be divisible by 16")
    if n % 32 != 0:
        raise ValueError(f"n={n} must be divisible by 32")

    torch.manual_seed(seed)

    x = torch.randn(m, k, device=device, dtype=dtype)
    weight_q = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    delta_base = (torch.rand(n, device=device, dtype=torch.bfloat16) * 0.2 + 0.01)
    num_groups = k // group_size
    e = torch.randint(
        -8,
        8,
        (n, num_groups),
        device=device,
        dtype=torch.int8,
    )

    weight_scale_grouped = _group_weight_scale(
        delta_base=delta_base,
        e=e,
        residual_bits=residual_bits,
    )

    return x, weight_q, delta_base, e, weight_scale_grouped


def run_reference_loop(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale_grouped: torch.Tensor,
    group_size: int,
):
    import torch_npu

    m, k = x.shape
    n = weight_q.shape[0]
    num_groups = k // group_size
    acc = torch.zeros(m, n, device=x.device, dtype=torch.float32)
    partials: list[torch.Tensor] = []

    weight_q_grouped = weight_q.view(n, num_groups, group_size)

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size
        x_group = x[:, start:end].contiguous()
        x_q_group, x_scale_group = torch_npu.npu_dynamic_quant(x_group)
        w_q_group = weight_q_grouped[:, group_idx, :].transpose(0, 1).contiguous()
        out_group = torch_npu.npu_quant_matmul(
            x_q_group,
            w_q_group,
            weight_scale_grouped[:, group_idx].contiguous(),
            pertoken_scale=x_scale_group,
            bias=None,
            output_dtype=x.dtype,
        )
        partials.append(out_group)
        acc.add_(out_group.to(torch.float32))

    return acc.to(x.dtype), partials


def run_k_group_kernel(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale_grouped: torch.Tensor,
    group_size: int,
    scale_override: torch.Tensor | None = None,
    block_dim: int = 8,
):
    import torch_npu

    m, k = x.shape
    num_groups = k // group_size
    x_q_packed, x_scale_grouped = _dynamic_quant_by_group(x.contiguous(), group_size)
    weight_nz = _pack_grouped_weight_to_nz(weight_q, group_size)
    group_list = torch.tensor([m], dtype=torch.int64, device=x.device)
    out = quant_matmul_grouped(
        x_int8=x_q_packed,
        weight_nz_int8=weight_nz,
        scale_f32=(
            scale_override
            if scale_override is not None
            else weight_scale_grouped.transpose(0, 1).contiguous().to(torch.float32)
        ),
        pertoken_scale_f32=x_scale_grouped.reshape(-1).contiguous().to(torch.float32),
        group_list=group_list,
        group_axis=1,
        k_group_size=group_size,
        k_group_num=num_groups,
        block_dim=block_dim,
    )
    return out.to(x.dtype)


def summarize_diff(ref: torch.Tensor, test: torch.Tensor) -> None:
    diff = (ref.to(torch.float32) - test.to(torch.float32)).abs()
    print(f"ref shape: {tuple(ref.shape)}, dtype: {ref.dtype}")
    print(f"test shape: {tuple(test.shape)}, dtype: {test.dtype}")
    print(f"mean abs diff: {float(diff.mean().cpu()):.8f}")
    print(f"max  abs diff: {float(diff.max().cpu()):.8f}")
    print(f"ref mean abs : {float(ref.abs().to(torch.float32).mean().cpu()):.8f}")
    print(f"test mean abs: {float(test.abs().to(torch.float32).mean().cpu()):.8f}")


def diagnose_diff(ref: torch.Tensor, test: torch.Tensor, topk: int = 8) -> None:
    ref_f = ref.to(torch.float32)
    test_f = test.to(torch.float32)
    finite = torch.isfinite(test_f)
    print(f"test finite: {int(finite.sum().cpu())}/{test.numel()}")
    if not bool(finite.all().cpu()):
        bad = (~finite).nonzero()
        first_bad = bad[0].tolist()
        print(f"first non-finite index: {first_bad}")
        bad_rows = torch.unique(bad[:, 0])[:16].tolist()
        bad_cols = torch.unique(bad[:, 1])[:16].tolist()
        print(f"non-finite rows sample: {bad_rows}")
        print(f"non-finite cols sample: {bad_cols}")

    safe_test = torch.where(finite, test_f, torch.zeros_like(test_f))
    diff = (ref_f - safe_test).abs()
    row_max = diff.max(dim=1).values
    col_max = diff.max(dim=0).values
    top_rows = torch.topk(row_max, k=min(topk, row_max.numel())).indices.tolist()
    top_cols = torch.topk(col_max, k=min(topk, col_max.numel())).indices.tolist()
    print(f"top diff rows: {top_rows}")
    print(f"top diff cols: {top_cols}")

    flat = diff.flatten()
    values, indices = torch.topk(flat, k=min(topk, flat.numel()))
    rows = indices // diff.shape[1]
    cols = indices % diff.shape[1]
    for rank, (value, row, col) in enumerate(zip(values, rows, cols)):
        row_i = int(row.cpu())
        col_i = int(col.cpu())
        print(
            f"top {rank:02d}: row={row_i}, col={col_i}, "
            f"diff={float(value.cpu()):.8f}, "
            f"ref={float(ref_f[row_i, col_i].cpu()):.8f}, "
            f"test={float(test_f[row_i, col_i].cpu()):.8f}"
        )
    for block in (128, 256):
        if diff.shape[1] <= block:
            continue
        print(f"column-block mean/max diff, block={block}:")
        for start in range(0, diff.shape[1], block):
            end = min(start + block, diff.shape[1])
            block_diff = diff[:, start:end]
            print(
                f"  cols {start:04d}-{end - 1:04d}: "
                f"mean={float(block_diff.mean().cpu()):.8f}, "
                f"max={float(block_diff.max().cpu()):.8f}"
            )


def summarize_partials(
    partials: list[torch.Tensor],
    ref: torch.Tensor,
    test: torch.Tensor,
) -> None:
    print(f"num partial groups: {len(partials)}")
    for idx, part in enumerate(partials):
        mean_abs = float(part.abs().to(torch.float32).mean().cpu())
        print(f"group {idx:03d} partial mean abs: {mean_abs:.8f}")
    print(
        "final checksum:",
        float(ref.to(torch.float32).sum().cpu()),
        float(test.to(torch.float32).sum().cpu()),
    )


def run_single_group_kernel_checks(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale_grouped: torch.Tensor,
    group_size: int,
):
    import torch_npu

    m, k = x.shape
    n = weight_q.shape[0]
    num_groups = k // group_size
    weight_q_grouped = weight_q.view(n, num_groups, group_size)
    results: list[tuple[int, float, float]] = []

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size
        x_group = x[:, start:end].contiguous()
        x_q_group, x_scale_group = torch_npu.npu_dynamic_quant(x_group)
        w_q_group = weight_q_grouped[:, group_idx, :].contiguous()

        ref_partial = torch_npu.npu_quant_matmul(
            x_q_group,
            w_q_group.transpose(0, 1).contiguous(),
            weight_scale_grouped[:, group_idx].contiguous(),
            pertoken_scale=x_scale_group,
            bias=None,
            output_dtype=x.dtype,
        )

        weight_nz_group = _pack_grouped_weight_to_nz(w_q_group, group_size)
        scale_group = (
            weight_scale_grouped[:, group_idx]
            .unsqueeze(0)
            .contiguous()
            .to(torch.float32)
        )
        group_list = torch.tensor([m], dtype=torch.int64, device=x.device)
        test_partial = quant_matmul_grouped(
            x_int8=x_q_group.contiguous(),
            weight_nz_int8=weight_nz_group,
            scale_f32=scale_group,
            pertoken_scale_f32=x_scale_group.contiguous().to(torch.float32),
            group_list=group_list,
            group_axis=0,
            k_group_size=0,
            k_group_num=0,
        ).to(x.dtype)

        diff = (ref_partial.to(torch.float32) - test_partial.to(torch.float32)).abs()
        results.append(
            (
                group_idx,
                float(diff.mean().cpu()),
                float(diff.max().cpu()),
            )
        )

    return results


def run_k_axis_single_group_kernel_checks(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale_grouped: torch.Tensor,
    group_size: int,
):
    import torch_npu

    m, k = x.shape
    n = weight_q.shape[0]
    num_groups = k // group_size
    weight_q_grouped = weight_q.view(n, num_groups, group_size)
    results: list[tuple[int, float, float]] = []

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size
        x_group = x[:, start:end].contiguous()
        x_q_group, x_scale_group = torch_npu.npu_dynamic_quant(x_group)
        w_q_group = weight_q_grouped[:, group_idx, :].contiguous()

        ref_partial = torch_npu.npu_quant_matmul(
            x_q_group,
            w_q_group.transpose(0, 1).contiguous(),
            weight_scale_grouped[:, group_idx].contiguous(),
            pertoken_scale=x_scale_group,
            bias=None,
            output_dtype=x.dtype,
        )

        weight_nz_group = _pack_grouped_weight_to_nz(w_q_group, group_size)
        scale_group = (
            weight_scale_grouped[:, group_idx]
            .unsqueeze(0)
            .contiguous()
            .to(torch.float32)
        )
        group_list = torch.tensor([m], dtype=torch.int64, device=x.device)
        test_partial = quant_matmul_grouped(
            x_int8=x_q_group.contiguous(),
            weight_nz_int8=weight_nz_group,
            scale_f32=scale_group,
            pertoken_scale_f32=x_scale_group.contiguous().to(torch.float32),
            group_list=group_list,
            group_axis=1,
            k_group_size=group_size,
            k_group_num=1,
        ).to(x.dtype)

        diff = (ref_partial.to(torch.float32) - test_partial.to(torch.float32)).abs()
        results.append(
            (
                group_idx,
                float(diff.mean().cpu()),
                float(diff.max().cpu()),
            )
        )

    return results


def main() -> None:
    args = build_parser().parse_args()
    dtype = get_dtype(args.dtype)

    x, weight_q, delta_base, e, weight_scale_grouped = build_inputs(
        m=args.m,
        k=args.k,
        n=args.n,
        group_size=args.group_size,
        residual_bits=args.residual_bits,
        dtype=dtype,
        device=args.device,
        seed=args.seed,
    )

    print("input config:")
    print(f"  device={args.device}")
    print(f"  m={args.m}, k={args.k}, n={args.n}")
    print(f"  group_size={args.group_size}, residual_bits={args.residual_bits}")
    print(f"  dtype={dtype}, seed={args.seed}")
    print(f"  delta_base dtype={delta_base.dtype}, e dtype={e.dtype}")

    ref, partials = run_reference_loop(
        x=x,
        weight_q=weight_q,
        weight_scale_grouped=weight_scale_grouped,
        group_size=args.group_size,
    )
    test = run_k_group_kernel(
        x=x,
        weight_q=weight_q,
        weight_scale_grouped=weight_scale_grouped,
        group_size=args.group_size,
        block_dim=args.block_dim,
    )

    summarize_diff(ref, test)
    if args.diagnose:
        diagnose_diff(ref, test)
    if args.dump_partials:
        summarize_partials(partials, ref, test)
    if args.isolate_groups:
        print("isolated K-group checks:")
        base_scale = weight_scale_grouped.transpose(0, 1).contiguous().to(torch.float32)
        for group_idx, ref_partial in enumerate(partials):
            scale_override = torch.zeros_like(base_scale)
            scale_override[group_idx].copy_(base_scale[group_idx])
            test_partial = run_k_group_kernel(
                x=x,
                weight_q=weight_q,
                weight_scale_grouped=weight_scale_grouped,
                group_size=args.group_size,
                block_dim=args.block_dim,
                scale_override=scale_override,
            )
            diff = (ref_partial.to(torch.float32) - test_partial.to(torch.float32)).abs()
            print(
                f"group {group_idx:03d} isolated mean abs diff: "
                f"{float(diff.mean().cpu()):.8f}, max abs diff: "
                f"{float(diff.max().cpu()):.8f}"
            )
    if args.check_single_group:
        print("single-group kernel checks:")
        for group_idx, mean_diff, max_diff in run_single_group_kernel_checks(
            x=x,
            weight_q=weight_q,
            weight_scale_grouped=weight_scale_grouped,
            group_size=args.group_size,
        ):
            print(
                f"group {group_idx:03d} mean abs diff: {mean_diff:.8f}, "
                f"max abs diff: {max_diff:.8f}"
            )
    if args.check_k_axis_single_group:
        print("K-axis single-group kernel checks:")
        for group_idx, mean_diff, max_diff in run_k_axis_single_group_kernel_checks(
            x=x,
            weight_q=weight_q,
            weight_scale_grouped=weight_scale_grouped,
            group_size=args.group_size,
        ):
            print(
                f"group {group_idx:03d} mean abs diff: {mean_diff:.8f}, "
                f"max abs diff: {max_diff:.8f}"
            )


if __name__ == "__main__":
    main()
