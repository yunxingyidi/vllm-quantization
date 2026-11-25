# QuantMatmulExpertKGrouped

## 双 AIV epilogue 实验分支

`GROUP_QUANT_GMM2_DUAL_AIV=1` 让默认 MIX 1:2 内核的两个 AIV 分别处理
一个输出 tile 的前后半段行。INT32 GM ring、逐组 scale、FP32 累加和 BF16
输出语义沿用原路径。空行消费者仍发送完成信号，AIC 继续等待整个 group 后复用 slot。
该分支默认启用，可设为0对照；local-int32 key 1 使用独立 epilogue，不受此开关影响。

```bash
bash custom_op/scripts/build_groupquant_moe_fused.sh
GROUP_QUANT_GMM2_DUAL_AIV=1 bash run.sh
```

同一二进制的旧/新路径需要分别启动进程，不能在已经创建 executor 或捕获 ACL Graph
后修改环境变量。原生 W8A8 对照及独立 FP64 golden：

```bash
source env_vllm_ascend.sh
ASCEND_RT_VISIBLE_DEVICES=4 python -m tests.custom_op.quant_matmul_expert_k_grouped.benchmark_backends \
  --dual-aiv 1 --m 257 --routing sparse --save-output /tmp/gmm2_dual.pt
ASCEND_RT_VISIBLE_DEVICES=4 python -m tests.custom_op.quant_matmul_expert_k_grouped.benchmark_backends \
  --dual-aiv 0 --m 257 --routing sparse --save-output /tmp/gmm2_legacy.pt
ASCEND_RT_VISIBLE_DEVICES=4 python -m tests.custom_op.quant_matmul_expert_k_grouped.benchmark_backends \
  --backend native --m 81920 --perf-only --graph --result /tmp/gmm2_native.json
```

benchmark 的两个后端使用相同 qweight、activation、per-token scale、路由和 BF16
可表示的 weight scale 数值。custom 的 scale 存储为 FP32；原生 BF16 输出按本机
CANN 约束传 BF16 scale。权重打包、golden 与输入量化不计入时间；`--graph` 报告图
replay 的同步 wall-clock 延迟，`--perf-only` 明确跳过 golden。多 K-group 仅支持
custom 对照。基准会将原始随机 scale 先舍入到 BF16，不能用此结果代替任意 FP32
scale 的完整精度覆盖。

独立的 registry-invoke GroupQuant GMM2 算子。该算子一次处理全部 expert，避免
`Smoothquant/fuse_moe/moe_mlp.py` 中逐 expert 的 Python 循环，不修改原有
`custom_op/quant_matmul_k_grouped` 算子。

## 接口

```python
torch.ops._C_ascend.quant_matmul_expert_k_grouped(
    x_int8,          # packed [G*M, S] or ND [M, G, S], int8
    weight_nz_int8,  # [E * G * N * group_size], int8 packed NZ
    group_list,      # [E], int64 expert token counts
    weight_scale,    # [E * G, N], float32
    x_scale,         # [M], float32, native per-token activation scale
) -> y              # [M, N], bfloat16 (native kernel output)
```

其中 `E` 为 expert 数，`G` 为 K-group 数。`group_list` 使用每 expert token 数，
不是累计 token 数。activation 使用 per-token 量化，因此所有 K group 复用同一个
`x_scale[m]`；只有 weight scale 沿 K 轴按 group 区分。

### 直接 ND 输入

rank3 `[M,G,S]` 直接引用 GMM1 连续 `[M,K]` 输出，Matmul 的 A 行跨度
为完整 K、B 为 S。多组预量化模型路径默认使用 ND，无需优化环境变量。
单组、动态量化输入和 local-int32 实验继续使用旧 packed 契约。
`GROUP_QUANT_GMM2_ND_INPUT=0` 可在新进程中恢复旧重排作对照。
新增契约需要成套重建算子包及 Torch adapter。

精度检查使用 `tests.custom_op.quant_matmul_expert_k_grouped.test_nd_input`；
性能基准可选 `--input-layout nd`，旧重排总耗时选
`--input-layout packed --include-pack`。这些参数只影响 custom 后端。
详细数据见 [ND 输入报告](../../docs/GMM2_ND_INPUT_RESULTS.md)。

## 构建

```bash
bash custom_op/scripts/build_quant_matmul_expert_k_grouped.sh
source custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash
```

构建脚本同时生成 registry 自定义算子包、安装到仓库内隔离目录，并构建 PyTorch
adapter，不会覆盖原 `quant_matmul_k_grouped`。

## GMM2 接入

`Smoothquant/fuse_moe/moe_mlp.py` 默认仍走原逐 expert fallback。显式开启：

```bash
export GROUP_QUANT_USE_EXPERT_GMM2_OP=1
```

`env_vllm_ascend.sh` also sets `GROUP_QUANT_CUSTOM_OPAPI_PATH` to this
project's absolute `libcust_opapi.so` path. The Torch adapter uses that path
directly, avoiding a SONAME collision with the custom-op library bundled in
newer vLLM-Ascend wheels.

如 adapter 不在默认构建目录，可通过
`GROUP_QUANT_EXPERT_GMM2_LIBRARY=/absolute/path/to/adapter.so` 指定。新路径在首次
调用时预打包静态 W2 权重和 scale，之后按 tensor 地址和版本复用；缓存容量默认 128，
可用 `GROUP_QUANT_EXPERT_GMM2_CACHE_SIZE` 调整。

## 精度与性能对比

测试以原 `quant_matmul_k_grouped` 的逐 expert 调用作为 fallback，并使用相同的
per-token 量化 activation、NZ 权重和 weight scale。旧 fallback 仍要求 `[G,M]`
scale buffer，测试只为适配 ABI 重复同一个 `[M]` token scale，不执行独立的
per-group activation quant：

```bash
python -m tests.custom_op.quant_matmul_expert_k_grouped.test_quant_matmul_expert_k_grouped
```

计时包含 NPU warmup，并在计时区间前后调用 `torch.npu.synchronize()`。静态权重打包
不计入 custom-op steady-state 延迟。

910B3 实测（表中历史结果为 FP16 输出版本；BF16 原生输出以 ST 当前结果为准）：

| M | K | N | E | group_size | custom | fallback | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 512 | 256 | 4 | 128 | 0.060 ms | 0.961 ms | 15.98x |
| 32 | 512 | 256 | 2 | 512 | 0.065 ms | 0.435 ms | 6.71x |
| 128 | 4096 | 4096 | 8 | 128 | 0.419 ms | 2.237 ms | 5.34x |

原生 per-token scale ABI 在以下 shape 上重新验证：

| M | K | N | E | group_size | custom | fallback | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 512 | 256 | 4 | 128 | 0.067742 ms | 1.081569 ms | exact match |
| 33 | 256 | 1024 | 8 | 16 | 0.117994 ms | 2.237907 ms | exact match |
| 257 | 768 | 2048 | 128 | 32 | 2.567644 ms | 34.797443 ms | exact match |

原生 BF16 输出版本（2026-07-26，910B3）重新验证：

| M | K | N | E | group_size | custom | fallback | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 512 | 256 | 4 | 128 | 0.083831 ms | 1.147555 ms | exact match, 13.689x |
| 257 | 768 | 2048 | 128 | 32 | 2.526443 ms | 39.553624 ms | exact match, 15.656x |

BF16 epilogue 保持 FP32 累加，并使用 `CAST_RINT` 一次写回 BF16；Python
仅在非 BF16 调用方请求兼容 dtype 时执行后置转换。

### Qwen3-30B-A3B GMM2 调优（2026-08-06，910B3）

整网 profiling 中 `QuantMatmulExpertKGrouped` 占算子总耗时约 36.4%，其中主调用
shape 为 `M=81920, K=384, N=2048, E=128, group_size=128`。原配置仅启动 8 个
AIC，主 shape 的 AIC MAC 占比约 8.3%，AIC/AIV scalar 占比分别约 40.5%/41.8%。

将切分从 `8 cores, base=128x256x128, parall=8` 调整为
`20 cores, base=64x512x128, parall=4` 后：

| shape | 原配置 | 新配置 | 加速比 |
|---|---:|---:|---:|
| `81920x384 @ 384x2048, E=128` | 7.518669 ms | 3.011737 ms | 2.497x |
| `16384x384 @ 384x2048, E=128` | 1.696837 ms | 1.449097 ms | 1.171x |

主 shape 的独立 ST 使用 `M=257, K=384, N=2048, E=128` 复核精度，结果通过
`rtol=2e-2, atol=2e-2`。`parall=4` 与 `parall=8` 性能相同（约 3.012 ms），但
workspace 减半；`baseK=64` 没有额外收益，因此保持一个 K-group 对应完整 128。

相同 batch6、ACL Graph、6 windows 的端到端结果从 `5994.65 tok/s` 提升到
`6520.94 tok/s`（`+8.78%`），PPL 保持 `8.246581`。相对最初 GMM1 group256 的
`4434.70 tok/s`，当前 GMM1+GMM2 调优后的累计提升为 `47.04%`；相对 BF16
`9586.80 tok/s` 已达到 `68.02%`，剩余吞吐差距约 `1.47x`。

### GMM2 单组与 scale 复用（2026-08-06，910B3）

`VectorComputeExpertKGrouped` 原先对每个 K-group 重复加载、广播相同的 per-token
activation scale。将其保留在 UB 并跨 group 复用后，`group_size=128` 主 shape 从
`3.018301 ms` 降至 `2.686629 ms`（`+12.34%`），端到端从 `6520.94` 小幅提升至
`6534.07 tok/s`，PPL 保持 `8.246581`。

GMM2 的 TP-local K 为384。新增 `GROUP_QUANT_GMM2_GROUP_SIZE` 独立覆盖后，设置为
`0`（完整 local K）使 GMM2 使用一个384 group，而 GMM1 仍使用独立的2048 group：

| GMM2 group | 主 shape | 小 shape | 20-window PPL | 20-window throughput |
|---:|---:|---:|---:|---:|
| 128 | 2.686629 ms | 1.296462 ms | 8.656475 | 6555.26 tok/s |
| 384 | 1.038984 ms | 0.505468 ms | 8.640936 | 6771.14 tok/s |

完整 K-group 将 GMM2 主 shape 加速 `2.59x`，20-window 整网提升 `3.29%`，且 PPL
略有改善，因此设为默认。重新扫描8/12/16/20核和 M/N/K tile 后，20核、
`64x512x128` 仍为最优。尝试在 UB 中跨 M 子块缓存 weight scale 仅带来约0.4%
收益且出现异步覆盖导致的精度错误，已完整回退。

同硬件、TP=2、batch=6、ACL Graph、20-window 的 BF16 基线为 `9540.91 tok/s`
（profile run；无 profile 的重复基线为 `9663.36 tok/s`）。当前量化默认配置为
`6771.14 tok/s`，约为 BF16 的 `70.97%`（按 profile 基线）至 `70.07%`（按无 profile
基线），剩余端到端差距约 `1.41x`–`1.43x`。后续优先继续压缩 GMM1/GMM2 的核内
标量与 FixPipe 开销；`DynamicQuant -> GMM1` 虽可融合，但其独立开销不足以单独弥合该差距。

### GMM1 核数复扫（2026-08-06，910B3）

在 full-K `M=8192, K=2048, N=1536, E=128` 的独立基准中，GMM1 的 8/12/16/20/24/28/32
核延迟为 `0.950/0.687/0.568/0.522/0.513/0.517/0.516 ms`。因此将默认核数从20更新为24；
更高核数开始出现调度开销，未保留。

同一 20-window 端到端配置（TP=2、batch=6、ACL Graph、GMM2 full local-K）中，GMM1
20 核为 `6771.14 tok/s`，24 核为 `6821.60 tok/s`（`+0.75%`），PPL 均为 `8.640936`。
