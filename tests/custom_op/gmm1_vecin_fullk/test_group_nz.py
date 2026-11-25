"""Run grouped-NZ precision cases; compare modes from independent processes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tests.custom_op.gmm1_vecin_fullk.benchmark_epilogue import main as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-nz", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("tp2_m65", 65, 768, 4, 64, "hotspot"),
        ("group128_boundaries", 257, 768, 8, 16, "uniform"),
        ("tp4_sparse", 257, 384, 16, 64, "sparse"),
        ("k16_m17", 17, 128, 2, 128, "hotspot"),
        ("two_groups", 33, 768, 4, 2, "hotspot"),
        ("tp2_m17", 17, 768, 4, 64, "hotspot"),
        ("tp1_fallback", 7, 1536, 2, 64, "hotspot"),
        ("tp8_fallback", 17, 192, 4, 64, "uniform"),
        ("single_group", 33, 768, 4, 1, "uniform"),
    ]
    original_argv = sys.argv
    try:
        for name, m, full_n, experts, groups, routing in cases:
            sys.argv = [
                original_argv[0], "--m", str(m), "--full-n", str(full_n),
                "--experts", str(experts), "--groups", str(groups),
                "--routing", routing, "--nz", "1",
                "--group-nz", str(args.group_nz), "--group-tile-m", "32",
                "--warmup", "1", "--iters", "1", "--repeats", "1", "--graph",
                "--result", str(args.output_dir / f"{name}.json"),
                "--save-output", str(args.output_dir / f"{name}.pt"),
            ]
            benchmark()
            print(name, "PASS independent golden", flush=True)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
