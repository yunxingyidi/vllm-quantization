"""Validate both activation layouts; run each dual-AIV mode in a fresh process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu

from tests.custom_op.quant_matmul_expert_k_grouped.benchmark_backends import golden, make_counts
from tests.custom_op.quant_matmul_expert_k_grouped.test_quant_matmul_expert_k_grouped import (
    build_inputs,
    load_torch_adapter,
    run_custom,
)


def main():
    parser = argparse.ArgumentParser(description="Validate GMM2 ND and packed input")
    parser.add_argument("--dual-aiv", choices=(0, 1), type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["GROUP_QUANT_GMM2_DUAL_AIV"] = str(args.dual_aiv)
    torch_npu.npu.config.allow_internal_format = True
    load_torch_adapter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("one_tail", 1, 384, 2048, 4, 32, "hotspot"),
        ("tp2_tail", 65, 384, 2048, 8, 32, "hotspot"),
        ("sparse_n_tail", 257, 384, 1056, 16, 32, "sparse"),
        ("tp4", 33, 192, 2048, 4, 32, "uniform"),
        ("single", 17, 384, 128, 4, 384, "uniform"),
        ("s16_odd_groups", 19, 48, 96, 4, 16, "uniform"),
    ]
    results = []
    for name, m, k, n, experts, group_size, routing in cases:
        xq, wq, wnz, _, ws, sx = build_inputs(m, k, n, experts, group_size, 1)
        counts = make_counts(m, experts, routing)
        device_counts = counts.to(xq.device)
        nd = xq.view(k // group_size, m, group_size).permute(1, 0, 2).contiguous()
        packed_output = run_custom(xq, wnz, device_counts, ws, sx).cpu()
        nd_output = run_custom(nd, wnz, device_counts, ws, sx).cpu()
        reference = golden(xq, wq, counts, ws, sx)
        torch.testing.assert_close(packed_output, reference, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(nd_output, reference, rtol=2e-2, atol=2e-2)
        assert torch.equal(packed_output, nd_output), name
        assert torch.isfinite(nd_output).all(), name
        torch.save(
            {"packed": packed_output, "nd": nd_output, "golden": reference},
            args.output_dir / f"{name}.pt",
        )
        results.append({
            "case": name, "m": m, "k": k, "n": n, "experts": experts,
            "group_size": group_size, "routing": routing, "dual_aiv": args.dual_aiv,
            "golden": "PASS", "bitwise_equal": True,
        })
        print(name, "PASS golden + bitwise", flush=True)

    # Reject a logical group count inconsistent with the scales before launch.
    try:
        run_custom(nd[:, :-1], wnz, device_counts, ws, sx)
    except RuntimeError as exc:
        assert "group dimension" in str(exc), str(exc)
        print("bad_group_count PASS rejection", flush=True)
    else:
        raise AssertionError("mismatched ND group count accepted")
    (args.output_dir / "precision_matrix.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
