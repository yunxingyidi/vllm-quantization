"""Compare quantized MoE routing against per-token quantize then route.

Uses real npu_moe_init_routing_v2, including returned expanded scales, and
times complete pipelines under ACL Graph. Does not modify production dispatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu


def make_input(m, h, k, experts, mode, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, h, generator=g, dtype=torch.float32).to(torch.bfloat16)
    if m >= 4:
        x[0] = 0
        x[1] *= 0.01
        x[2] *= 20
        x[3, :8] = torch.tensor([-32, -8, -1, 0, 0.1, 1, 8, 32], dtype=x.dtype)
    if mode == 'hotspot':
        ids = torch.arange(k, dtype=torch.int32).repeat(m, 1)
    else:
        base = torch.randint(experts, (m, 1), generator=g, dtype=torch.int32)
        ids = (base + torch.arange(k, dtype=torch.int32)) % experts
    return x.npu(), ids.npu()


def router(x, ids, experts, scale=None, quant_mode=-1, row_idx_type=0):
    return torch_npu.npu_moe_init_routing_v2(
        x, ids, scale=scale, active_num=ids.numel(), expert_num=experts,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[0, experts], quant_mode=quant_mode, row_idx_type=row_idx_type)


def separate(x, ids, experts, flat_ids):
    q, scale = torch_npu.npu_dynamic_quant(x)
    _, destination, counts, _ = router(x[:, :1].contiguous(), ids, experts, quant_mode=-1)
    # expanded_row_idx maps each input (token, top-k) slot to its output
    # position. Invert that permutation to obtain the original token per row.
    reverse = torch.empty_like(destination)
    reverse.scatter_(0, destination.to(torch.int64), flat_ids)
    token = torch.div(reverse, ids.shape[1], rounding_mode='floor').to(torch.int64)
    return q.index_select(0, token), destination, counts, scale.index_select(0, token)


def token_map(destination, k, flat_ids):
    reverse = torch.empty_like(destination)
    reverse.scatter_(0, destination.to(torch.int64), flat_ids)
    return torch.div(reverse, k, rounding_mode='floor').to(torch.int64)


def separate_permute(x, ids, experts, flat_ids):
    q, scale = torch_npu.npu_dynamic_quant(x)
    permuted, destination = torch_npu.npu_moe_token_permute(q, ids.to(torch.int64))
    _, _, counts, _ = router(x[:, :1].contiguous(), ids, experts, quant_mode=-1)
    token = token_map(destination, ids.shape[1], flat_ids)
    return permuted, destination, counts, scale.index_select(0, token)


def separate_double_permute(x, ids, experts):
    q, scale = torch_npu.npu_dynamic_quant(x)
    ids64 = ids.to(torch.int64)
    permuted, destination = torch_npu.npu_moe_token_permute(q, ids64)
    expanded_scale, _ = torch_npu.npu_moe_token_permute(scale.reshape(-1, 1), ids64)
    _, _, counts, _ = router(x[:, :1].contiguous(), ids, experts, quant_mode=-1)
    return permuted, destination, counts, expanded_scale.reshape(-1)


def separate_rowidx1(x, ids, experts):
    q, scale = torch_npu.npu_dynamic_quant(x)
    permuted, destination = torch_npu.npu_moe_token_permute(q, ids.to(torch.int64))
    _, source, counts, _ = router(x[:, :1].contiguous(), ids, experts,
                                  quant_mode=-1, row_idx_type=1)
    token = torch.div(source, ids.shape[1], rounding_mode='floor').to(torch.int64)
    return permuted, destination, counts, scale.index_select(0, token)


def check(m, h, k, experts, mode, seed):
    x, ids = make_input(m, h, k, experts, mode, seed)
    q, scales = torch_npu.npu_dynamic_quant(x)
    fused = router(x, ids, experts, quant_mode=1)
    flat_ids = torch.arange(m * k, dtype=torch.int32, device=x.device)
    separated = separate(x, ids, experts, flat_ids)
    via_permute = separate_permute(x, ids, experts, flat_ids)
    via_two_permute = separate_double_permute(x, ids, experts)
    via_rowidx1 = separate_rowidx1(x, ids, experts)
    torch.npu.synchronize()
    fx, fi, fc, fs = (z.cpu() for z in fused)
    sx, si, sc, ss = (z.cpu() for z in separated)
    px, pi, pc, ps = (z.cpu() for z in via_permute)
    dx, di, dc, ds = (z.cpu() for z in via_two_permute)
    rx, ri, rc, rs = (z.cpu() for z in via_rowidx1)
    qc, ids_cpu, scales_cpu = q.cpu(), ids.cpu(), scales.cpu().reshape(-1)
    # Stable sort over flattened expert selections gives the independent
    # expert-major reference. Check exact flattened assignment and scale order.
    order = torch.argsort(ids_cpu.reshape(-1), stable=True)
    token = torch.div(order, k, rounding_mode='floor')
    expected_q = qc[token]
    expected_scale = scales_cpu[token]
    counts = torch.bincount(ids_cpu.reshape(-1).long(), minlength=experts)
    diff = (fx.to(torch.int16) - sx.to(torch.int16)).abs()
    result = {
        'm': m, 'h': h, 'k': k, 'experts': experts, 'routing': mode,
        'output_rows': fx.shape[0], 'counts_equal': bool(torch.equal(fc.long(), sc.long())),
        'counts_match_reference': bool(torch.equal(fc.long(), counts)),
        'index_equal': bool(torch.equal(fi, si)),
        'prequant_q_matches_stable_expert_reference': bool(torch.equal(sx, expected_q)),
        'prequant_scale_matches_stable_expert_reference': bool(torch.equal(ss, expected_scale)),
        'permute_path_matches_gather': bool(torch.equal(px, sx) and torch.equal(pi, si)
                                           and torch.equal(pc, sc) and torch.equal(ps, ss)),
        'two_permute_path_matches_gather': bool(torch.equal(dx, sx) and torch.equal(di, si)
                                               and torch.equal(dc, sc) and torch.equal(ds, ss)),
        'rowidx1_path_matches_gather': bool(torch.equal(rx, sx) and torch.equal(ri, si)
                                            and torch.equal(rc, sc) and torch.equal(rs, ss)),
        'fused_q_matches_prequant': bool(torch.equal(fx, sx)),
        'q_mismatch_count': int((diff != 0).sum()), 'q_max_abs_diff': int(diff.max()),
        'fused_scale_bitwise_matches_prequant': bool(torch.equal(fs, ss)),
        'scale_max_abs_diff': float((fs-ss).abs().max()),
        'scale_allclose_1e-6': bool(torch.allclose(fs, ss, rtol=1e-6, atol=1e-8)),
        'finite_scales': bool(torch.isfinite(fs).all() and torch.isfinite(ss).all()),
    }
    if not all(result[k] for k in ('counts_equal', 'counts_match_reference', 'index_equal',
                                    'prequant_q_matches_stable_expert_reference',
                                    'prequant_scale_matches_stable_expert_reference',
                                    'permute_path_matches_gather', 'two_permute_path_matches_gather',
                                    'rowidx1_path_matches_gather', 'finite_scales')):
        raise AssertionError(json.dumps(result, indent=2))
    return result


def graph(fn, warmup):
    for _ in range(warmup):
        out = fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = fn()
    return g, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device-label', default='unknown')
    p.add_argument('--m', type=int, nargs='+', default=[2048, 10240])
    p.add_argument('--h', type=int, default=2048)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--experts', type=int, default=128)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--result', type=Path, required=True)
    a = p.parse_args()
    checks = [check(m, a.h, a.topk, a.experts, mode, 42)
              for m, mode in [(1, 'hotspot'), (17, 'uniform'), (65, 'hotspot'), (257, 'uniform')]]
    print('CORRECTNESS', json.dumps(checks), flush=True)
    cases = []
    for m in a.m:
        x, ids = make_input(m, a.h, a.topk, a.experts, 'uniform', 42)
        q, s = torch_npu.npu_dynamic_quant(x)
        flat_ids = torch.arange(m*a.topk, dtype=torch.int32, device=x.device)
        _, destination, _, _ = router(x[:, :1].contiguous(), ids, a.experts, quant_mode=-1)
        source_token = token_map(destination, a.topk, flat_ids)
        funcs = {
            'fused_quant_dispatch': lambda: router(x, ids, a.experts, quant_mode=1),
            'quantize_only': lambda: torch_npu.npu_dynamic_quant(x),
            'carrier_route_only': lambda: router(x[:, :1].contiguous(), ids, a.experts, quant_mode=-1),
            'invert_mapping_only': lambda: token_map(destination, a.topk, flat_ids),
            'gather_int8_only': lambda: q.index_select(0, source_token),
            'gather_scale_only': lambda: s.index_select(0, source_token),
            'int8_moe_permute_only': lambda: torch_npu.npu_moe_token_permute(q, ids.to(torch.int64)),
            'scale_moe_permute_only': lambda: torch_npu.npu_moe_token_permute(s.reshape(-1, 1), ids.to(torch.int64)),
            'prequant_then_dispatch': lambda: separate(x, ids, a.experts, flat_ids),
            'prequant_then_permute': lambda: separate_permute(x, ids, a.experts, flat_ids),
            'prequant_then_two_permutes': lambda: separate_double_permute(x, ids, a.experts),
            'prequant_then_rowidx1': lambda: separate_rowidx1(x, ids, a.experts),
            'bf16_dispatch': lambda: router(x, ids, a.experts, quant_mode=-1),
        }
        graphs = {name: graph(fn, a.warmup) for name, fn in funcs.items()}
        timings = {name: [] for name in funcs}
        for rep in range(a.repeats):
            names = list(funcs)
            if rep % 2:
                names.reverse()
            for name in names:
                g, out = graphs[name]
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(a.iters):
                    g.replay()
                torch.npu.synchronize()
                timings[name].append((time.perf_counter()-start)*1000/a.iters)
                assert out is not None
        result = {'m': m, 'h': a.h, 'topk': a.topk, 'experts': a.experts,
                  'timings': {name: {'median_ms': statistics.median(v), 'samples_ms': v}
                              for name, v in timings.items()}}
        cases.append(result)
        print('CASE', json.dumps(result), flush=True)
        del graphs, funcs, x, ids, q, s
    payload = {'device_label': a.device_label, 'checks': checks, 'cases': cases,
               'warmup': a.warmup, 'iters': a.iters, 'repeats': a.repeats,
               'timing': 'ACL Graph replay synchronized wall clock; alternating order',
               'torch': torch.__version__, 'torch_npu': torch_npu.__version__}
    a.result.parent.mkdir(parents=True, exist_ok=True)
    a.result.write_text(json.dumps(payload, indent=2)+'\n')

if __name__ == '__main__':
    main()
