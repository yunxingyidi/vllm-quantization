"""Compare GMM1 epilogues in independent processes with identical quantized inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from Smoothquant.fuse_moe.moe_mlp import _load_gmm1_vecin_fullk_op
from tests.custom_op.quant_matmul_expert_k_grouped.benchmark_backends import make_counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("custom", "native"), default="custom")
    p.add_argument("--nz", type=int, choices=(0, 1), default=0)
    p.add_argument("--group-nz", type=int, choices=(0, 1), default=None)
    p.add_argument("--group-tile-m", type=int, choices=(4, 8, 16, 32), default=32)
    p.add_argument("--nz-accum", type=int, choices=(0, 1), default=None)
    p.add_argument("--tile-m", type=int, choices=(4, 8, 16, 32), default=32)
    p.add_argument("--m", type=int, default=257)
    p.add_argument("--full-n", type=int, default=768)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--routing", choices=("uniform", "sparse", "hotspot"), default="uniform")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--perf-only", action="store_true")
    p.add_argument("--graph", action="store_true")
    p.add_argument("--save-output", type=Path)
    p.add_argument("--result", type=Path)
    a = p.parse_args()
    if min(a.m, a.experts, a.groups, a.warmup, a.iters, a.repeats) <= 0:
        p.error("shapes and iteration counts must be positive")
    if 2048 % a.groups or (2048 // a.groups) % 16 or a.full_n % 64 or not 0 < a.full_n <= 1536:
        p.error("unsupported K-group or local N")
    if a.backend != "custom" and a.groups != 1:
        p.error("native comparison requires one full-K group")
    os.environ["GROUP_QUANT_GMM1_NZ_EPILOGUE"] = str(a.nz)
    if a.nz_accum is not None:
        os.environ["GROUP_QUANT_GMM1_NZ_ACCUM"] = str(a.nz_accum)
    os.environ["GROUP_QUANT_GMM1_NZ_BASE_M"] = str(a.tile_m)
    if a.group_nz is not None:
        os.environ["GROUP_QUANT_GMM1_GROUP_NZ"] = str(a.group_nz)
    os.environ["GROUP_QUANT_GMM1_GROUP_BASE_M"] = str(a.group_tile_m)
    torch.manual_seed(a.seed)
    torch_npu.npu.config.allow_internal_format = True
    qx_cpu = torch.randint(-127, 128, (a.m, 2048), dtype=torch.int8)
    qw_cpu = torch.randint(-127, 128, (a.experts, a.full_n, 2048), dtype=torch.int8)
    sx_cpu = torch.rand(a.m) * 0.005 + 0.001
    sw_cpu = torch.rand(a.experts, a.groups, a.full_n) * 0.0005 + 0.0001
    qx = qx_cpu.npu()
    sx = sx_cpu.npu()
    packed_scale = torch_npu.npu_trans_quant_param(sw_cpu.reshape(-1).npu()).reshape_as(sw_cpu)
    counts = make_counts(a.m, a.experts, a.routing)
    ends = counts.cumsum(0).npu()
    effective_sw = (packed_scale.cpu().bitwise_and(0xffffffff)
                    .to(torch.int32).view(torch.float32)).reshape_as(sw_cpu)
    if a.backend == "custom":
        weights = [torch_npu.npu_format_cast(w.T.contiguous().npu(), 29) for w in qw_cpu]
        _load_gmm1_vecin_fullk_op()

        def fn():
            return torch.ops._C_ascend.gmm1_vecin_fullk(qx, weights, packed_scale, sx, ends)
    else:
        native_weight = torch_npu.npu_format_cast(qw_cpu.transpose(1, 2).contiguous().npu(), 29)
        native_scale = effective_sw[:, 0].contiguous().npu()

        def fn():
            q, scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                qx, native_weight, ends, native_scale, sx)
            return q, scale

    output, scales = fn()
    torch.npu.synchronize()
    precision = "skipped"
    if not a.perf_only:
        # Compare the actual FixPipe parameters in the ABI. TransQuantParam
        # may canonicalize mantissa bits; the pre-pack random FP32 tensor is
        # not necessarily the scale the kernel receives.
        # Match the deployed FixPipe FP16 weight-scaled intermediate, then
        # use CPU FP32 for token scaling and SwiGLU. Operand order is deliberate.
        pre = torch.zeros((a.m, a.full_n), dtype=torch.float32)
        start = 0
        for expert, count in enumerate(counts.tolist()):
            end = start + count
            for group in range(a.groups):
                if count:
                    k0 = group * (2048 // a.groups)
                    k1 = (group + 1) * (2048 // a.groups)
                    dots = qx_cpu[start:end, k0:k1].double() @ qw_cpu[expert, :, k0:k1].double().T
                    scaled = dots.float() * effective_sw[expert, group]
                    pre[start:end] += (scaled.clamp(-65504, 65504).half().float()
                                      if a.backend == "custom" else scaled)
            start = end
        pre *= sx_cpu[:, None]
        h = a.full_n // 2
        post = torch.nn.functional.silu(pre[:, :h]) * pre[:, h:]
        ref_scale = post.abs().amax(1).clamp_min(1e-6) / 127
        ref = torch.round(post / ref_scale[:, None]).to(torch.int8)
        actual = output.cpu()
        actual_scale = scales.cpu()
        assert torch.isfinite(actual_scale).all()
        torch.testing.assert_close(actual_scale, ref_scale, rtol=2e-3, atol=1e-7)
        max_lsb = (actual.short() - ref.short()).abs().max().item()
        assert max_lsb <= 1, f"INT8 max LSB error={max_lsb}"
        precision = f"passed: {a.backend} golden, INT8 max LSB={max_lsb}"
    if a.save_output:
        a.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": output.cpu(), "scale": scales.cpu(), "counts": counts}, a.save_output)
    for _ in range(a.warmup):
        fn()
    torch.npu.synchronize()
    graph = None
    if a.graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = fn()
        call = graph.replay
    else:
        call = fn
    samples = []
    for _ in range(a.repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(a.iters):
            call()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / a.iters)
    if graph is not None:
        assert captured[0].numel() == a.m * (a.full_n // 2)
    result = {"backend": a.backend, "nz": a.nz, "nz_accum_requested": a.nz_accum,
              "group_nz_requested": a.group_nz, "group_tile_m": a.group_tile_m,
              "requested_tile_m": a.tile_m, "m": a.m,
              "full_n": a.full_n, "experts": a.experts, "groups": a.groups,
              "routing": a.routing, "seed": a.seed, "graph": a.graph,
              "precision": precision, "median_ms": statistics.median(samples),
              "samples_ms": samples, "warmup": a.warmup, "iters": a.iters}
    from tests.performance.profile_operator_pair import capture_operator
    capture_operator(fn)
    if a.result:
        a.result.parent.mkdir(parents=True, exist_ok=True)
        a.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
