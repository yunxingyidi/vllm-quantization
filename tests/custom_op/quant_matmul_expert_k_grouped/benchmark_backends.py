"""Same-input GMM2 correctness and timing; run each tiling mode in a fresh process.

Packing and golden calculation are outside timed regions unless --include-pack
explicitly includes activation packing with the custom call.
Native INT8 and BF16 comparisons use one full-K group here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from tests.custom_op.quant_matmul_expert_k_grouped.test_quant_matmul_expert_k_grouped import (
    build_inputs,
    load_torch_adapter,
    run_custom,
)


def make_counts(m: int, experts: int, routing: str) -> torch.Tensor:
    counts = torch.zeros(experts, dtype=torch.int64)
    if routing == "hotspot":
        counts[experts // 2] = m
    else:
        active = experts if routing == "uniform" else max(1, experts // 8)
        # Leave zero-count experts interspersed, including before a hotspot.
        indices = torch.arange(active) * (experts // active)
        counts[indices] = m // active
        counts[indices[:m % active]] += 1
    return counts


def golden(x_q, weight_q, counts, weight_scale, x_scale):
    """CPU float64 dot reference, including each group's distinct scale."""
    groups = weight_scale.shape[1]
    m = x_q.shape[0] // groups
    group_size = x_q.shape[1]
    qx = x_q.cpu().reshape(groups, m, group_size).double()
    qw = weight_q.cpu().double()
    ws = weight_scale.cpu().double()
    sx = x_scale.cpu().double()
    y = torch.empty((m, weight_scale.shape[-1]), dtype=torch.float64)
    start = 0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        if count:
            acc = torch.zeros((count, y.shape[1]), dtype=torch.float64)
            for group in range(groups):
                k0 = group * group_size
                dots = qx[group, start:end] @ qw[expert, :, k0:k0 + group_size].T
                acc += dots * ws[expert, group]
            y[start:end] = acc * sx[start:end, None]
        start = end
    return y.to(torch.bfloat16)


def golden_bf16(x, weight, counts):
    """Reference for the rounded BF16 operands used by the native backend."""
    x = x.cpu().double()
    weight = weight.cpu()
    y = torch.empty((x.shape[0], weight.shape[2]), dtype=torch.float64)
    start = 0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        if count:
            y[start:end] = x[start:end] @ weight[expert].double()
        start = end
    return y.to(torch.bfloat16)


def measure(fn, warmup: int, iters: int, repeats: int, capture: bool):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    graph = None
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured_output = fn()
        fn = graph.replay
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iters)
    # Keep captured tensors alive for every replay.
    if graph is not None:
        assert captured_output.numel() > 0
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("custom", "native", "bf16"), default="custom")
    parser.add_argument("--dual-aiv", type=int, choices=(0, 1), default=0)
    parser.add_argument("--m", type=int, default=257)
    parser.add_argument("--k", type=int, default=384)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=384)
    parser.add_argument("--routing", choices=("uniform", "sparse", "hotspot"), default="uniform")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale-values", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--custom-weight-scale-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--input-layout", choices=("packed", "nd"), default="packed")
    parser.add_argument("--include-pack", action="store_true",
                        help="Time ND-to-group-major packing with the packed custom call")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--save-output", type=Path)
    args = parser.parse_args()
    if min(args.m, args.k, args.n, args.experts, args.group_size,
           args.warmup, args.iters, args.repeats) <= 0:
        parser.error("shapes and iteration counts must be positive")
    if args.backend != "custom" and args.group_size != args.k:
        parser.error("BF16 and native INT8 comparisons require a single full-K group")
    if args.backend != "custom" and args.dual_aiv:
        parser.error("--dual-aiv applies only to the custom backend")
    if args.backend == "native" and args.scale_values != "bf16":
        parser.error("native BF16 output requires BF16-representable scale values")

    if args.backend != "custom" and (args.input_layout != "packed" or args.include_pack):
        parser.error("input layout options apply only to custom")
    if args.include_pack and args.input_layout != "packed":
        parser.error("--include-pack requires packed layout")

    # Env-based host tiling is read before the first custom call. Never toggle
    # it in-process: executor/tiling caches need not include env in their keys.
    os.environ["GROUP_QUANT_GMM2_DUAL_AIV"] = str(args.dual_aiv)
    torch_npu.npu.config.allow_internal_format = True
    xq, wq, wnz, _, ws, sx = build_inputs(
        args.m, args.k, args.n, args.experts, args.group_size, args.seed)
    # Production full-K delta_base is BF16. Use identical representable scale
    # values in both backends: CANN requires BF16 scales for BF16 native output.
    # Custom still consumes its FP32 ABI; this conversion changes storage only.
    if args.scale_values == "bf16":
        ws = ws.to(torch.bfloat16).to(torch.float32)
    counts = make_counts(args.m, args.experts, args.routing)
    device_counts = counts.to(xq.device)
    if args.backend == "custom":
        load_torch_adapter()
        if args.custom_weight_scale_dtype == "bf16":
            ws = ws.to(torch.bfloat16)
        groups = args.k // args.group_size
        # Reconstruct the row-major producer output outside the timed region.
        xnd = xq.view(groups, args.m, args.group_size).permute(1, 0, 2).contiguous()
        if args.input_layout == "nd":
            fn = lambda: run_custom(xnd, wnz, device_counts, ws, sx)
        elif args.include_pack:
            fn = lambda: run_custom(
                xnd.permute(1, 0, 2).contiguous().view(groups * args.m, args.group_size),
                wnz, device_counts, ws, sx)
        else:
            fn = lambda: run_custom(xq, wnz, device_counts, ws, sx)
    elif args.backend == "native":
        # Exactly the same qweight values; only the static NZ ABI differs.
        native_weight = torch_npu.npu_format_cast(wq.transpose(1, 2).contiguous(), 29)
        native_scale = ws[:, 0].to(torch.bfloat16).contiguous()

        def fn():
            return torch_npu.npu_grouped_matmul(
                x=[xq], weight=[native_weight], scale=[native_scale],
                per_token_scale=[sx], group_list=device_counts,
                split_item=3, group_type=0, group_list_type=1,
                output_dtype=torch.bfloat16,
            )[0]
    else:
        # Match the serving BF16 GMM2 ABI with dense [E, K, N] weights.
        # Dequantize the shared inputs before timing or capturing the graph.
        x_bf16 = (xq.float() * sx.reshape(-1, 1)).to(torch.bfloat16)
        weight_bf16 = (wq.float() * ws[:, 0, :, None]).to(torch.bfloat16)
        weight_bf16 = weight_bf16.transpose(1, 2).contiguous()

        def fn():
            return torch_npu.npu_grouped_matmul(
                x=[x_bf16], weight=[weight_bf16], group_list=device_counts,
                split_item=2, group_type=0, group_list_type=1,
            )[0]

    output = fn()
    torch.npu.synchronize()
    precision = "skipped"
    if not args.perf_only:
        if args.backend == "bf16":
            reference = golden_bf16(x_bf16, weight_bf16, counts)
        else:
            reference = golden(xq, wq, counts, ws, sx)
        actual = output.cpu()
        assert actual.dtype == torch.bfloat16
        assert actual.shape == reference.shape
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
        precision = "passed: rtol=2e-2, atol=2e-2 against independent float64 golden"
    if args.save_output:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"output": output.cpu(), "counts": counts, "seed": args.seed}, args.save_output)
    result = {
        "backend": args.backend, "dual_aiv": args.dual_aiv,
        "input_layout": args.input_layout, "include_pack": args.include_pack,
        "shape": {"m": args.m, "k": args.k, "n": args.n,
                  "experts": args.experts, "group_size": args.group_size},
        "routing": args.routing, "seed": args.seed, "graph": args.graph,
        "scale_values": args.scale_values,
        "weight_scale_dtype": str(
            native_scale.dtype if args.backend == "native" else ws.dtype
        ),
        "precision": precision, "warmup": args.warmup, "iters": args.iters,
        "tiling_env": {key: value for key, value in os.environ.items()
                       if key.startswith("GROUP_QUANT_GMM2_")},
        **measure(fn, args.warmup, args.iters, args.repeats, args.graph),
    }
    from tests.custom_op.profile_operator_pair import capture_operator
    capture_operator(fn)
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
