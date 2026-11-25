# Gmm1VecInFullK design

## Scheduling

A registered 1:1 MIX block owns one logical `(expert, M-tile)` at a time.
Logical tiles are enumerated from the cumulative routing list and distributed
cyclically over the selected Cube cores. Each tile has at most 16 rows. The
AIC runs the Matmul KFC service; the paired AIV owns the UB buffers and runs
the epilogue.

Gate and up form one continuous `local_N` plane. For
`256 < local_N <= 768`, one synchronous full-N request traverses the Matmul
service's internal Cube N blocks. Other shapes use runtime N tiles no wider
than 256, or `local_N/2` for a narrow half. K groups remain serial inside the
owning MIX block, avoiding cross-core reduction and any GM accumulator.
`SetOrgShape(M,N,2048,2048,N)` preserves the original ND A and NZ B full-K
strides while `SetSingleShape` limits each reduction to one group.

## Local memory and Cube output

The Matmul C type is `TPosition::VECIN`, `CubeFormat::NZ`, `half`. For TP=2
and TP=4, guarded synchronous `IterateAll(LocalTensor)` transfers the complete
local N plane to the paired AIV UB in one KFC request. TP=1 and TP=8 use
`Iterate()` + `GetTensorC(LocalTensor)` for each runtime N tile. The kernel
registers the `Matmul` object with `REGIST_MATMUL_OBJ`; AIC never allocates
UB. On dav-2201 this transfer is an FP16 FixPipe conversion. Raw INT32 VECIN
output is unsupported, and there is no GM accumulator workspace.

Representative UB use for M=16 is approximately:

| Shape | Matmul N | Approximate bytes |
|---|---:|---:|
| TP=1, full N=1536 | runtime/UB-selected | runtime M-dependent |
| TP=2, full N=768 | 768 | below 183 KiB |
| TP=4, full N=384 | 384 | below 183 KiB |
| TP=8, full N=192 | 192 | below 183 KiB |

The largest allocation remains below the 910B3 ordinary vector allocation
limit of 184 KiB. Buffers are sized from the selected runtime Matmul N and
aligned output width instead of reserving every temporary at TP=1 maximum.

Packed `[E,G,N]` channel scales are consumed by FixPipe through
`SetQuantVector`, avoiding both raw-dot FP16 overflow and an AIV-side scale
multiply. The epilogue casts each contiguous NZ FP16 tile once to FP32 NZ.
The first group scatters 16-column fragments with `Copy`; subsequent groups
scatter and accumulate with their required `Add`. This eliminates FP16
identity Muls and the FP16 ND staging tensor. The token scale is applied once
after every K group, followed by `silu(gate)*up`, row reduction, and INT8
quantization.

The host selects M and N jointly under a 183-KiB UB model. Per-channel work
uses an adaptive M tile (`clamp(ceil(totalM / aicNum), 4, 16)`) to expose more
cores. K-group work remains at M=16 because splitting a synchronous 64-group
KFC loop increased launch/weight traffic enough to dominate the parallelism
gain.

FP32 gate/up, SwiGLU, reductions, and INT8 VECOUT rows use a
64-element-aligned stride. Final row-wise DMA removes that padding; for
example, TP=8 maps a 96-wide logical row to a 128-wide UB row.

The only GM writes are `y` and `y_scale`. The synchronous Matmul call supplies
FIX-to-vector completion; channel-scale DMA is owned by the Matmul/FixPipe
path. `V_MTE3` events guard final output DMA, and PIPE_V barriers order
dependent vector primitives and buffer reuse.

The full-N request optimization and its measured result are documented in
相关约束已直接收敛在本文档中。
