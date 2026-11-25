from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


def _import_enable_custom_op():
    try:
        from vllm_ascend.utils import enable_custom_op
        return enable_custom_op
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent / "vllm-ascend"
        import sys

        sys.path.insert(0, str(repo_root))
        from vllm_ascend.utils import enable_custom_op
        return enable_custom_op


enable_custom_op = _import_enable_custom_op()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correctness and latency check for grouped_matmul_swiglu_quant_weight_nz_tensor_list GroupQuant path."
    )
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--atol-scale", type=float, default=5e-3)
    parser.add_argument("--rtol-scale", type=float, default=5e-3)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--stage56-boundary-probe", action="store_true")
    return parser


def parse_scale_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def build_cumsum_group_list(m: int, experts: int) -> torch.Tensor:
    base = torch.zeros(experts, dtype=torch.int64)
    for token_idx in range(m):
        base[token_idx % experts] += 1
    return base.cumsum(dim=0)


def make_inputs(
    m: int,
    k: int,
    n: int,
    experts: int,
    group_size: int,
    scale_dtype: torch.dtype,
    seed: int,
):
    if k % group_size != 0:
        raise ValueError(f"k={k} must be divisible by group_size={group_size}")
    if n % 2 != 0:
        raise ValueError(f"n={n} must be even for SwiGLU, got n={n}")

    torch.manual_seed(seed)
    g = k // group_size

    x = torch.randint(-127, 128, (m, k), dtype=torch.int8)
    weight = torch.randint(-127, 128, (experts, k, n), dtype=torch.int8)
    weight_scale = (torch.rand(experts, g, n, dtype=torch.float32) * 0.09 + 0.01).to(scale_dtype)
    x_scale = (torch.rand(m, dtype=torch.float32) * 0.09 + 0.01)
    group_list = build_cumsum_group_list(m, experts)
    return x, weight, weight_scale, x_scale, group_list


def reference_group_quant_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_size: int,
    gate_first: bool = True,
    reuse_first_group_scale: bool = False,
    return_swiglu: bool = False,
):
    experts = weight.shape[0]
    n = weight.shape[-1]
    output = torch.empty((x.shape[0], n // 2), dtype=torch.int8)
    output_scale = torch.empty((x.shape[0],), dtype=torch.float32)
    swiglu_output = torch.empty((x.shape[0], n // 2), dtype=torch.float32) if return_swiglu else None
    accum_output = torch.empty((x.shape[0], n), dtype=torch.float32) if return_swiglu else None

    start = 0
    group_list_cpu = group_list.cpu().tolist()
    for expert_idx in range(experts):
        end = int(group_list_cpu[expert_idx])
        if end <= start:
            continue
        expert_x = x[start:end].to(torch.float32)
        expert_x_scale = x_scale[start:end].to(torch.float32)
        expert_weight = weight[expert_idx].to(torch.float32)
        expert_weight_scale = weight_scale[expert_idx].to(torch.float32)

        acc = torch.zeros((end - start, n), dtype=torch.float32)
        for group_idx in range(expert_weight_scale.shape[0]):
            k_start = group_idx * group_size
            k_end = k_start + group_size
            partial = expert_x[:, k_start:k_end] @ expert_weight[k_start:k_end, :]
            scale_group_idx = 0 if reuse_first_group_scale else group_idx
            partial = partial * expert_weight_scale[scale_group_idx].unsqueeze(0)
            partial = partial * expert_x_scale.unsqueeze(1)
            acc += partial

        first, second = acc.chunk(2, dim=-1)
        if gate_first:
            gate, up = first, second
        else:
            gate, up = second, first
        swiglu = F.silu(gate) * up
        scale = swiglu.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
        q = torch.round(swiglu / scale.unsqueeze(1)).clamp_(-128, 127).to(torch.int8)

        output[start:end] = q
        output_scale[start:end] = scale
        if swiglu_output is not None:
            swiglu_output[start:end] = swiglu
            accum_output[start:end] = acc
        start = end

    if swiglu_output is not None:
        return output, output_scale, swiglu_output, accum_output
    return output, output_scale


def run_npu_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
):
    weight_nz = [torch_npu.npu_format_cast(weight[idx].npu(), 29) for idx in range(weight.shape[0])]
    weight_scale_list = [weight_scale[idx].reshape(-1).npu().contiguous() for idx in range(weight_scale.shape[0])]

    y_q, y_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
        x=x.npu(),
        weight=weight_nz,
        weight_scale=weight_scale_list,
        x_scale=x_scale.npu(),
        group_list=group_list.npu(),
    )
    valid_m = int(group_list[-1].item())
    # Debug: print raw scale output and x_scale buffer tail
    print(f"  [debug] y_scale_raw[:4]={y_scale[:min(4, y_scale.shape[0])].cpu().tolist()}")
    print(f"  [debug] y_scale_raw[-4:]={y_scale[max(0,y_scale.shape[0]-4):].cpu().tolist()}")
    print(f"  [debug] y_scale_shape={y_scale.shape}")
    return y_q[:valid_m].cpu(), y_scale[:valid_m].cpu()


def benchmark_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    weight_nz = [torch_npu.npu_format_cast(weight[idx].npu(), 29) for idx in range(weight.shape[0])]
    weight_scale_list = [weight_scale[idx].reshape(-1).npu().contiguous() for idx in range(weight_scale.shape[0])]
    x_npu = x.npu()
    x_scale_npu = x_scale.npu()
    group_list_npu = group_list.npu()

    for _ in range(warmup):
        torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
            x=x_npu,
            weight=weight_nz,
            weight_scale=weight_scale_list,
            x_scale=x_scale_npu,
            group_list=group_list_npu,
        )
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
            x=x_npu,
            weight=weight_nz,
            weight_scale=weight_scale_list,
            x_scale=x_scale_npu,
            group_list=group_list_npu,
        )
    torch.npu.synchronize()
    elapsed_s = time.perf_counter() - start
    return elapsed_s / iters


def main() -> None:
    args = build_parser().parse_args()
    if args.perf_only:
        args.skip_correctness = True

    torch_npu.npu.config.allow_internal_format = True
    enable_custom_op()

    x, weight, weight_scale, x_scale, group_list = make_inputs(
        m=args.m,
        k=args.k,
        n=args.n,
        experts=args.experts,
        group_size=args.group_size,
        scale_dtype=parse_scale_dtype(args.scale_dtype),
        seed=args.seed,
    )

    print("config:")
    print(f"  M={args.m}, K={args.k}, N={args.n}, experts={args.experts}")
    print(f"  group_size={args.group_size}, group_num={args.k // args.group_size}")
    print(f"  scale_dtype={weight_scale.dtype}, group_list={group_list.tolist()}")

    if not args.skip_correctness:
        ref_result = reference_group_quant_swiglu(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
            group_size=args.group_size,
            return_swiglu=args.stage56_boundary_probe,
        )
        if args.stage56_boundary_probe:
            ref_q, ref_scale, ref_swiglu, ref_accum = ref_result
        else:
            ref_q, ref_scale = ref_result
        alt_ref_q, alt_ref_scale = reference_group_quant_swiglu(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
            group_size=args.group_size,
            gate_first=False,
        )
        reuse_g0_q, reuse_g0_scale = reference_group_quant_swiglu(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
            group_size=args.group_size,
            reuse_first_group_scale=True,
        )
        npu_q, npu_scale = run_npu_op(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
        )

        if args.stage56_boundary_probe:
            if args.m != 2 or args.n != 256:
                raise ValueError("stage56 boundary probe requires M=2 and N=256")
            probe = npu_q.contiguous().view(torch.float32).reshape(args.m, -1)
            target_row = 1
            src0 = probe[target_row, 0:8]
            src1 = probe[target_row, 8:16]
            swiglu_exit = probe[target_row, 16:24]
            marker = probe[target_row, 24:32]
            ref_src0 = ref_accum[target_row, 128 + 16:128 + 24]
            ref_src1 = ref_accum[target_row, 16:24]
            reference = ref_swiglu[target_row, 16:24]
            print("stage56 row1 SwiGLU input/output probe (channels 16:24):")
            print(f"  vector_mte_marker={marker.tolist()}")
            print(f"  src0_up={src0.tolist()}")
            print(f"  src0_up_reference={ref_src0.tolist()}")
            print(f"  src0_up_max_diff={float((src0 - ref_src0).abs().max()):.8f}")
            print(f"  src1_gate={src1.tolist()}")
            print(f"  src1_gate_reference={ref_src1.tolist()}")
            print(f"  src1_gate_max_diff={float((src1 - ref_src1).abs().max()):.8f}")
            print(f"  swiglu_exit={swiglu_exit.tolist()}")
            print(f"  reference={reference.tolist()}")
            print(f"  ch19_exit={float(swiglu_exit[3]):.8f}")
            print(f"  exit_ref_max_diff={float((swiglu_exit - reference).abs().max()):.8f}")
            return

        q_equal = torch.equal(npu_q, ref_q)
        scale_diff = (npu_scale - ref_scale).abs()

        print("correctness:")
        print(f"  quant_output_equal={q_equal}")
        print(f"  scale_mean_abs_diff={float(scale_diff.mean()):.8f}")
        print(f"  scale_max_abs_diff={float(scale_diff.max()):.8f}")

        if not q_equal:
            mismatch = (npu_q != ref_q).sum().item()
            print(f"  quant_output_mismatch_count={mismatch}")
            print(f"  ref_scale_head={ref_scale[:min(4, ref_scale.numel())].tolist()}")
            print(f"  alt_ref_scale_head={alt_ref_scale[:min(4, alt_ref_scale.numel())].tolist()}")
            print(f"  reuse_g0_scale_head={reuse_g0_scale[:min(4, reuse_g0_scale.numel())].tolist()}")
            print(f"  npu_scale_head={npu_scale[:min(4, npu_scale.numel())].tolist()}")
            print(f"  ref_q_row0_head={ref_q[0, :min(16, ref_q.shape[1])].tolist()}")
            print(f"  alt_ref_q_row0_head={alt_ref_q[0, :min(16, alt_ref_q.shape[1])].tolist()}")
            print(f"  reuse_g0_q_row0_head={reuse_g0_q[0, :min(16, reuse_g0_q.shape[1])].tolist()}")
            print(f"  npu_q_row0_head={npu_q[0, :min(16, npu_q.shape[1])].tolist()}")

        torch.testing.assert_close(
            npu_scale,
            ref_scale,
            atol=args.atol_scale,
            rtol=args.rtol_scale,
        )
        if not q_equal:
            raise AssertionError("quantized output mismatch")

    avg_s = benchmark_op(
        x=x,
        weight=weight,
        weight_scale=weight_scale,
        x_scale=x_scale,
        group_list=group_list,
        warmup=args.warmup,
        iters=args.iters,
    )
    total_gemm_ops = 2.0 * args.m * args.k * args.n
    tops = total_gemm_ops / avg_s / 1e12
    toks_per_s = args.m / avg_s

    print("performance:")
    print(f"  avg_latency_ms={avg_s * 1e3:.3f}")
    print(f"  throughput_tokens_per_s={toks_per_s:.2f}")
    print(f"  effective_tops={tops:.3f}")


if __name__ == "__main__":
    main()
