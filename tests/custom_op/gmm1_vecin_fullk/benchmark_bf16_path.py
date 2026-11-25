"""Performance-only BF16 MoE GMM1 path and standalone quantization probes.

These probes measure real operators, not internal phases of the fused INT8 kernel.
Use benchmark_epilogue for the INT8 comparison with matching M/K/N/E/routing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from tests.custom_op.quant_matmul_expert_k_grouped.benchmark_backends import make_counts


def measure(fn, args):
    for _ in range(args.warmup):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = fn()
    samples = []
    for _ in range(args.repeats):
        torch.npu.synchronize()
        begin = time.perf_counter()
        for _ in range(args.iters):
            graph.replay()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - begin) * 1000 / args.iters)
    # Keep captured outputs and their storage alive until the graph is finished.
    assert captured is not None
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=16384)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--full-n", type=int, default=768)
    p.add_argument("--projection", choices=("gmm1", "gmm2"), default="gmm1")
    p.add_argument("--routing", choices=("uniform", "sparse", "hotspot"), default="uniform")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--result", type=Path, required=True)
    args = p.parse_args()
    if min(args.m, args.experts, args.k, args.full_n, args.warmup, args.iters, args.repeats) <= 0 or args.full_n % 2:
        p.error("positive dimensions and even full-n are required")
    torch.manual_seed(args.seed)
    torch_npu.npu.config.allow_internal_format = True
    x = (torch.randn(args.m, args.k) * .1).bfloat16().npu()
    weight = (torch.randn(args.experts, args.full_n, args.k) * .01).bfloat16().npu()
    weight = torch_npu.npu_format_cast(weight.transpose(1, 2).contiguous(), 29)
    counts = make_counts(args.m, args.experts, args.routing)
    ends = counts.cumsum(0).npu()

    def gmm1():
        return torch_npu.npu_grouped_matmul(
            x=[x], weight=[weight], split_item=2, group_list_type=0,
            group_type=0, group_list=ends)[0]

    gate_up = gmm1()
    post = torch_npu.npu_swiglu(gate_up) if args.projection == "gmm1" else gate_up
    torch.npu.synchronize()
    assert gate_up.shape == (args.m, args.full_n)
    assert post.shape == (args.m, args.full_n // 2 if args.projection == "gmm1" else args.full_n)
    assert torch.isfinite(post).all().item()

    def chain():
        return torch_npu.npu_swiglu(gmm1())

    def chain_quant():
        return torch_npu.npu_dynamic_quant(chain())

    probes = {
        "bf16_gmm1": gmm1,
        "bf16_swiglu_on_precomputed_gate_up": lambda: torch_npu.npu_swiglu(gate_up),
        "input_per_token_dynamic_quant": lambda: torch_npu.npu_dynamic_quant(x),
        "output_per_token_dynamic_quant_on_precomputed_activation": lambda: torch_npu.npu_dynamic_quant(post),
        "bf16_gmm1_plus_swiglu": chain,
        "bf16_gmm1_plus_swiglu_plus_output_quant": chain_quant,
    }
    if args.projection == "gmm2":
        probes = {"bf16_gmm2": gmm1}
    results = {}
    for name, fn in probes.items():
        results[name] = measure(fn, args)
        print(name, results[name], flush=True)
    result = {
        "m": args.m, "k": args.k, "full_n": args.full_n, "experts": args.experts,
        "projection": args.projection,
        "routing": args.routing, "seed": args.seed, "dtype": "bfloat16", "weight_format": "NZ/29",
        "counts": counts.tolist(), "graph": True, "warmup": args.warmup,
        "iters": args.iters, "repeats": args.repeats,
        "precision": "performance-only; shape and finite output checked; no mathematical golden",
        "note": "Separate operator probes are not fused INT8 phase timings; do not sum medians as chain latency.",
        "torch": torch.__version__, "torch_npu": torch_npu.__version__, "results": results,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
