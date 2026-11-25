# GMM1 VecIn Full-K kernel

This operator is a dedicated replacement candidate for the GroupQuant MoE
GMM1 hot path.  It deliberately does **not** replace the generic operator.

Model dispatch now prefers native fused GMM1 for supported single-group
weights. This custom operator remains the grouped/EPLB fallback and can be
selected with `GROUP_QUANT_GMM1_BACKEND=custom` before loading the model.
See [native GMM1 results](../../docs/NATIVE_GMM1_ACCELERATION_RESULTS.md).

## Supported contract

* activation: INT8 `[M, 2048]`, native FP32 per-token scale `[M]`;
* weights: per-expert INT8 NZ `[local_N / 32, 2048 / 16, 16, 32]`;
* weight scale: packed INT64 `[expert_num, K_group_num, local_N]`, produced
  once from FP32 scales by `torch_npu.npu_trans_quant_param`;
* routing: cumulative INT64 expert counts;
* result: INT8 `[M, local_N / 2]` and FP32 per-token result scale `[M]`.

`local_N` must be a multiple of 64 in `(0,1536]`. This covers the
Qwen3-30B-A3B local widths for TP=1/2/4/8: `1536/768/384/192`, with output
widths `768/384/192/96` respectively. Gate and up are traversed as one
continuous plane using `M<=16, N<=768, K=K_group_size` Matmul work, including
a runtime N tail. For
`256 < local_N <= 768`, one `IterateAll` request traverses all internal Cube
N blocks; TP=1 remains 256-wide tiled and TP=8 retains its compact narrow-N
path.

A default single-group path uses a batch NZ epilogue and adaptive `M<=32`.
It requires `K_group_num=1` and `(local_N/2)%64==0` (TP=1/2/4); other shapes
use the existing path. It processes the full local N in one request, applies
token scaling and SwiGLU in NZ, and converts only the half-width result to ND.
Lifetime-based UB reuse allows TP2 M32; TP1 automatically reduces M to fit.
The host queries actual UB size and reserves space for the Matmul service.
Set `GROUP_QUANT_GMM1_NZ_EPILOGUE=0` to use the legacy epilogue. The maximum M is controlled by
`GROUP_QUANT_GMM1_NZ_BASE_M` (default 32, bounded to 4..32). Both host and
kernel must be rebuilt together. `GROUP_QUANT_GMM1_LOG_TILING=1` reports the
actual selection. See [measured results](../../docs/GMM1_ACCELERATION_RESULTS.md).
`K_group_num` must divide 2048 and `K_group_size` must be 16-aligned. The
per-channel form is represented by `K_group_num=1`.
Each Cube result is declared
`TPosition::VECIN`; the vector epilogue reads it locally,
dequantizes with the input and channel scales, computes SwiGLU, dynamically
applies the token scale, quantizes the row, and writes only the ABI result
tensors to GM.  It must not
write an INT32 accumulator tensor to GM.

The dedicated dispatch is valid only for `swiglu_limit == 0`. `weight` and
weight is a dynamic tensor list with one item per expert. Scales for all
experts are packed into one tensor to keep a 128-expert invocation below the
runtime launcher input limit. All experts use the same `local_N` and group
count.

## Synchronization model

The 910B implementation is a registered 1:1 MIX kernel. The AIC runs the
Matmul KFC service and FixPipe, while its paired AIV owns the FP16 VECIN
buffer and completes both halves plus the vector epilogue. The generic GMM1
global AIV redistribution is intentionally not reused.

## Build and opt-in dispatch

Build the independent Torch adapter after the CANN custom-op package has been
installed:

```bash
cd custom_op/gmm1_vecin_fullk
VLLM_ASCEND_CSRC=/path/to/vllm-ascend/csrc python setup.py build_ext
```

For model deployment, build/install GMM1 and the expert GMM2 operator in one
package because both Torch adapters resolve the same `libcust_opapi.so`:

```bash
bash build.sh --ops=gmm1_vecin_fullk,quant_matmul_expert_k_grouped \
  --soc=ascend910b --pkg --vendor_name=custom
```

The model path is disabled by default. Enable it explicitly with:

```bash
export GROUP_QUANT_CUSTOM_OPAPI_PATH=/absolute/path/to/libcust_opapi.so
export GROUP_QUANT_USE_GMM1_VECIN_FULLK=1
```

`Smoothquant/fuse_moe/moe_mlp.py` loads the adapter lazily and dispatches only
the fused GMM1 branch. With the flag unset, the existing generic operator is
unchanged.

## Verification

1. Verify one expert / one `M=16` tile against the FP32 golden result. (done)
2. Verify expert-boundary, M-tail routing, FP16-overflow, and TP=1/2/4/8
   local widths. (done)
3. Add the Python dispatch gate, disabled by default. (done)
4. Run the GMM1 direct benchmark. (done)
5. Run the TP=2 20-window end-to-end model evaluation. (done)

On Ascend910B3, raw INT32 VECIN is unavailable. The kernel therefore applies
the packed per-channel weight scale in FixPipe before converting L0C to FP16.
This avoids the former raw-dot saturation at `65504` without writing an INT32
accumulator to GM. Token scaling and K-group accumulation remain FP32 on AIV.

The deterministic random-NZ case passes with FP32 scale MERE
`5.6315e-06`, MARE `2.81575e-05`, and 100% of INT8 outputs within one LSB.
The direct ACLNN benchmark (one warm-up, 50 independently constructed
executors) averages `64.0289 us` for the full `M=16` tile.

The seven-metric msprof capture reports `69.442 us` task duration,
`14.781 us` AIC time, `13.867 us` AIV time, 21.29% Cube utilization, and
0.30% vector bank conflict. No pipeline exceeds 80%; the dominant cost for
this small case is the 54.661 us (78.7%) task/head overhead, so the capture
does not justify labeling the kernel compute- or bandwidth-bound. These are
historical M16 results. The current M16384 NZ/M32 capture is archived in
[`docs/perf/round_002_nz_m32/summary.txt`](docs/perf/round_002_nz_m32/summary.txt);
its measurements and MIX parsing limitations are discussed in the native
GMM1 report linked above.

The TP=2 `--residual-group 32` model uses 64 K groups with local scales
`[128,64,768]`. It now dispatches to this operator successfully. The
20-window evaluation completed with PPL `17.014444`, 5100 predicted tokens,
76.850 s elapsed, and 66.62 tok/s. The grouped implementation accumulates
FP32 in UB. Its tuned group-outer schedule loads one complete scale row per
K group, traverses gate+up as one continuous full-N plane, and applies the
token scale only after all groups have accumulated. For TP=2 this reduces
Cube calls from four to three per K group and scale DMA instructions from
four to one. A V-to-MTE2 anti-dependency protects the reused scale buffer.

For the direct TP=2 `M=16`, `K_group_num=64`, `local_N=768` case, the same
50-iteration ACLNN benchmark improved from `795.825 us` to `623.332 us` and
`634.451 us` in two post-change runs (mean `628.892 us`, 21.0% lower). A
subsequent run from the combined GMM1+GMM2 deployment package measured
`614.364 us`.

### Full-N KFC request and adaptive-M optimization

The Matmul tiling now selects the widest runtime N tile that fits the 183-KiB
UB budget and uses local-output `IterateAll`. Full local-N execution is active
for TP=1/2/4/8; TP=8 no longer falls back to the per-tile `Iterate/GetTensorC`
path. The M tile is also runtime-selected from total M and the 20 AICs, with a
minimum tile of four. K-group paths retain M=16 because profiling showed that
splitting their synchronous KFC loop duplicates weight traffic and regresses
latency. All TP=1/2/4/8 random K-group and FP16-overflow tests pass. In the
earlier same-device comparison before the NZ epilogue change,
the TP=2 K-group32 benchmark moved from `611.444/614.887 us` (old package) to
`332.367/333.715/333.369/333.772 us` (new package): means `613.166 us` and
`333.306 us`, a 45.6% latency reduction (1.84x speedup). The unchanged TP=1
path measured `60.344/61.106 us` after the change.

The implementation rationale, UB sizing, failed TP=8 trial, verification
matrix, and raw measurements are recorded in
具体调度约束已合并到 [`docs/gmm1_vecin_fullk_design.md`](docs/gmm1_vecin_fullk_design.md)。

With FixPipe channel scaling enabled, K-group accumulation casts one complete
`[M_tile,N_tile]` FP16 result to FP32 per Cube call instead of issuing a Cast
and two barriers for every row. A same-device four-run comparison measured
`630.657 us` mean versus `646.188 us` for the row-wise version (2.4% lower).
An attempted asynchronous `Matmul::Iterate<false>` is intentionally not used:
DAV2201 VECIN FixPipe raised an MTE address-range exception, so the retained
`IterateAll` and `Iterate/GetTensorC` paths both keep a synchronous KFC
contract.

### NZ epilogue without identity Muls

The current epilogue casts the contiguous FP16 VECIN NZ tile once to FP32 NZ.
For the first K group, vector `Copy` scatters each 16-column fragment into the
row-major accumulation buffer; later groups use their required FP32 `Add`
directly for the same scatter. Therefore the old TP=2/K-group32 sequence of
3072 FP16 `Muls(x, 1)` instructions is completely removed, as is the FP16 ND
staging buffer.

DAV_2201 does not permit `vcopy` to read the Matmul VECIN tensor directly, and
CANN 8.5.1 exposes NZ-to-ND DMA only for UB-to-GM. Direct ND FixPipe was also
measured slower. The retained FP32 scatter keeps every intermediate local and
preserves the requested FP16 FixPipe behavior. Its same-device ACLNN results
are `50.906 us` for TP=1/per-channel, `440.782 us` for TP=2/K-group32, and
`312.026 us` for TP=8/K-group32. The identity-Mul-free TP=2 path remains 28.1%
faster than the original `613.166 us` full-N baseline, although it is slower
than the `333.306 us` identity-Mul implementation; this is the measured cost
of the supported VECIN-to-VECCALC layout path on DAV_2201.

The post-tuning TP=2 20-window regression preserves PPL `17.014444` and 5100
predicted tokens, completing in `73.480 s` at `69.68 tok/s`. This is 4.6%
higher throughput than the earlier `66.62 tok/s` run. This regression used
the generic GMM2 path to isolate GMM1 because the expert-GMM2 dummy-profile
input currently presents one weight expert with 128 routing counts.

The current model integration materializes the packed INT64 FixPipe scale at
weight-load time. `TransQuantParamV3` rejects the logical rank-3
`[E,K_group_num,N]` input on CANN 8.5.1, so loading flattens it for the
elementwise conversion and restores the logical view afterward. Production
forward no longer reconstructs FP32 K-group scales or launches
`npu_trans_quant_param` per layer. A TP=2, residual-group=32 regression with
custom GMM1 and expert-GMM2 completed dummy profiling, ACL graph capture, and
six 2048-token windows: PPL `8.223525`, 12,282 predicted tokens, 11.045 s,
and 1112.49 tok/s.

### Grouped NZ accumulation

For multiple K groups, full-N tiles with N<=768 and half-width divisible by
64 now accumulate the FP32 partials in NZ, converting to ND once after the
last group. Each group keeps its FP16 FixPipe rounding and the original
FP32 addition order. The original row epilogue then consumes the ND result.
The implementation reuses existing buffers and retains the M16 limit.
TP1/TP8 and unsupported widths use the original path automatically.

This preceding accumulation path remains available for comparison. First set
`GROUP_QUANT_GMM1_GROUP_NZ=0`, then compare `GROUP_QUANT_GMM1_NZ_ACCUM=0/1`
before starting new processes. The tiling
cache does not include environment variables, so changing this within an
existing process is not a valid comparison. Results and coverage are in
[the group32 report](../../docs/GROUP32_NZ_ACCUM_RESULTS.md).

### Streamed partial and grouped NZ epilogue (current default)

Eligible multi-group shapes (N<=768, half-width divisible by 64) now use a
single full FP32 NZ accumulator and a bounded 4096-float partial scratch.
After all groups, the existing NZ epilogue consumes the accumulator directly;
only its half-width SwiGLU result is converted to ND. This makes M32 fit in
176640 bytes at N768, below the 183 KiB UB budget. Weight storage, per-group
FixPipe FP16 scaling, FP32 addition order, and the operator ABI stay the same.

The host selects this path automatically (`nzEpilogue=2`). Unsupported shapes
and single-group calls retain their existing paths. For an independent-process
comparison, `GROUP_QUANT_GMM1_GROUP_NZ=0` restores the preceding M16 epilogue;
`GROUP_QUANT_GMM1_GROUP_BASE_M=16` isolates the tile-size effect while keeping
the new epilogue. No optimization variable is required for normal startup.
When reproducing the earlier `NZ_ACCUM=0/1` experiment, also disable GROUP_NZ.
See [the grouped M32 report](../../docs/GMM1_GROUP_NZ_M32_RESULTS.md).

### Reuse stable OrgShape

The grouped NZ path now retains OrgM/OrgN within each kernel invocation and
calls SetOrgShape only when they change. Tail tiles update the shape before
computing; every group still updates SingleShape, A/B pointers, and its packed
channel scale. The cache contains shape metadata only. Weight layout, UB
buffers, FixPipe rounding, group order, and outputs retain their contracts.

This defaults on for eligible grouped NZ shapes. Set
`GROUP_QUANT_GMM1_REUSE_ORG_SHAPE=0` before starting a new process for a
comparison. No optimization variable is required for normal startup. See
[the control-overhead report](../../docs/GROUP128_CONTROL_OVERHEAD_RESULTS.md).
