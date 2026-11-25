# Gmm1VecInFullK definition

## Contract

This is a shape-specialized GroupQuant MoE GMM1 operator for Ascend 910B3.
It accepts `x:int8[M,2048]`, `x_scale:float32[M]`, one FRACTAL_NZ int8
weight per expert with physical shape `[local_N/32,128,16,32]`, packed int64
FixPipe scales `weight_scale[E,G,local_N]`, and cumulative
`group_list:int64[E]`. The packed values encode the logical FP32 channel
scales through `torch_npu.npu_trans_quant_param`. It
returns `y:int8[M,local_N/2]` and
`y_scale:float32[M]`.

`local_N` is a multiple of 64 in `(0,1536]`; all experts in one invocation
have the same width. Qwen3-30B-A3B TP=1/2/4/8 uses
`local_N=1536/768/384/192`.

`weight` is a dynamic TensorList with `len(weight) == len(group_list) == E >=
1`; `weight_scale` is one contiguous tensor so 128 experts do not exceed the
launcher input limit. `G` divides 2048 and `K_group_size=2048/G` is a multiple
of 16. `G=1` is the per-channel case. This path is valid only when the
caller's `swiglu_limit` is zero. Scales are finite and positive.

For row `m` routed to expert `e`:

```
dot[m,g,n] = int32_dot(x[m,g*K_group_size:(g+1)*K_group_size],
                       logical_NZ(weight[e])[g*K_group_size:(g+1)*K_group_size,n])
cube[m,g,n] = fp16(dot[m,g,n] * logical_scale(weight_scale[e,g,n]))
deq[m,n] = x_scale[m] * sum_g(fp32(cube[m,g,n]))
H        = local_N / 2
up       = deq[m,0:H]
gate     = deq[m,H:local_N]
z        = silu(gate) * up
s[m]     = max(max(abs(z)), 1e-6) / 127
y[m,:]   = clamp(round_to_even(z / s[m]), -128, 127)
```

Let `g[-1]=0`; expert `e` owns rows `[g[e-1],g[e])`. Every expert is tiled
independently into `ceil((g[e]-g[e-1])/16)` tiles, so a tile never crosses an
expert boundary. Empty experts are valid. `group_list` is non-decreasing,
`0 <= g[e] <= M`, and `g[E-1] == M`. M=0 is rejected by tiling because no
kernel launch is useful for this dedicated hot path.

No int32 accumulator is part of the ABI and none may be written to GM.

Each `cube` group is intentionally FP16 on Ascend 910B3: its public FixPipe
API cannot materialize an INT32 L0C result in VECIN. Applying the channel
scale in FixPipe prevents realistic GroupQuant dots from saturating before
dequantization. This remains an approximate path because the scaled value is
rounded to FP16 before FP32 K-group accumulation.
