# Original-token quantization and top-k scatter

Two-stage local dispatch experiment for Ascend 910B/CANN 8.5.1:

1. Native nonquantized routing of a single-column carrier generates exact expert ordering, inverse mapping and counts.
2. One custom vector kernel owns each original token, computes its absmax/127 scale, rounds to INT8, and writes all top-k copies plus scale using the native mapping.

The prototype preserves current expert metadata and eliminates repeated quantization on expanded data. It does not fuse sorting into the custom kernel. The operator consumes a complete INT32 bijective `(token,slot)->destination` mapping and finite BF16 `[M,H]`; H must be a multiple of 64 and <=8192, top-k <=64. Full expert range only; the caller guarantees valid unique destinations. Per-core UB use is bounded to approximately `17*H + align(topk,8)*4 + 32` bytes, below the target DAV2201 UB capacity. It queries available vector core count and guards the input device at runtime. No CPU tensor fallback or output caching.

Build after sourcing `env_vllm_ascend.sh`:

```bash
bash custom_op/dispatch_quant_scatter/run.sh
```

On workstations with the local benchmark scripts in `tests/performance/`, the comparison can be run with:

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python tests/performance/benchmark_dispatch_scatter.py \
  --result tests/artifacts/dispatch_quant_scatter_20261031/check.json
```

Opt-in model experiment: `GROUP_QUANT_DISPATCH_SCATTER=1`. Currently integrated only with the legacy AllGather dispatcher, BF16 contiguous inputs, at least 2048 original tokens, no expert map, pre-existing token scale, router-weight-on-input, or EPLB. Other configurations retain the existing path. The final kernel matches native routing's FP32→FP16 truncation followed by FP16→INT8 rounding. Tested small and large inputs have bitwise-identical INT8 data, scales and metadata; 100-window PPL validation and timing limits are recorded in the performance report.

The build/kernel files were adapted from the provided CANN `add_custom` direct-invoke template; copyright/license notices are retained. Static shape checks and device allocation happen in the Torch binding. The kernel wrapper is launched directly on the current ACL stream after `stream(true)` drains prior host tasks and supports ACL Graph replay.
