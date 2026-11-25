# GroupQuant MatMul 与融合 MoE SwiGLU 算子

## 1. 项目概述

本目录实现了面向 Ascend 910B 的 GroupQuant MatMul 融合算子：

```text
grouped_matmul_swiglu_quant_weight_nz_tensor_list
```

该算子服务于 MoE MLP 的第一层计算，将以下操作融合在一个 Ascend custom op 中：

```text
Grouped Int8 MatMul
  + K 轴 GroupQuant 反量化累加
  + SwiGLU
  + Per-token Dynamic Quant
```

算子直接输出供下一层量化 MatMul 使用的 int8 activation 和 per-token scale，减少 expert
循环、kernel launch、中间 FP tensor 写回和重复数据搬运。

`custom_op/quant_group_matmul/` 保留独立直调参考实现，用于验证 M 轴分组、K 轴分组、
动态量化和不同流水策略。本目录即融合生产算子的源码目录：

```text
custom_op/grouped_matmul_swiglu_quant_weight_nz_tensor_list/
```

## 2. 算子功能

### 2.1 输入语义

- `x`：已经动态量化的 int8 activation，逻辑形状为 `[M, K]`。
- `weight`：每个 expert 一份 int8 NZ 权重，通过 tensor list 传入，逻辑形状为 `[K, N]`。
- `weightScale`：每个 expert 的 GroupQuant weight scale，形状为 `[G, N]`，支持
  `float32` 和 `float16`。
- `xScale`：输入 activation 的 per-token scale，形状为 `[M]`，dtype 为 `float32`。
- `groupList`：各 expert token 的累计结束位置，长度为 expert 数，最后一个值等于 `M`。
- `swigluLimit`：SwiGLU 输入裁剪上限；为 0 时不执行限幅。

其中：

```text
G = K / group_size
N 必须为偶数
group_size 必须满足 NZ MatMul 的对齐约束
```

### 2.2 输出语义

- `output`：SwiGLU 后重新量化的 int8 activation，形状为 `[M, N/2]`。
- `outputScale`：输出的 per-token dynamic quant scale，形状为 `[M]`，dtype 为 `float32`。

### 2.3 数学定义

对属于 expert `e` 的 token `m`，先沿 K 轴按 group 计算 int32 partial：

```text
partial[g, m, c] = sum_k x_int8[m, g, k] * weight_int8[e, g, k, c]
```

再应用 per-group weight scale 和 per-token activation scale：

```text
acc[m, c] = sum_g partial[g, m, c]
                    * weight_scale[e, g, c]
                    * x_scale[m]
```

将输出通道分成 up 和 gate 两部分。当前算子约定前半通道为 up、后半通道为 gate：

```text
up   = acc[:, :N/2]
gate = acc[:, N/2:]
y    = silu(gate) * up
```

最后按 token 动态量化：

```text
abs_max[m] = max(abs(y[m, :]))
output_scale[m] = max(abs_max[m], 1e-6) / 127
output[m, :] = round(y[m, :] / output_scale[m]).clamp(-128, 127).int8()
```

## 3. 实现方法

### 3.1 AIC：分组 Cube MatMul

输入 X 先按 K-group 重排为 `[G, M, group_size]`。AIC 对每个 expert 和 K-group 执行
int8 Cube MatMul，输出 int32 partial 到 workspace：

```text
[G, M, group_size] x [expert, K, N] -> [G, M, N]
```

权重保持原始 NZ 布局，不再额外复制和重排。MatMul 使用五参数 `SetOrgShape` 分别表达：

- A 的逻辑 K stride：`group_size`；
- B 在原始 NZ tensor 中的物理 K stride：`K`；
- C 的行 stride：`N`。

M/N block 由 AIC core 并行分发，支持 M 尾块和多个 N block。

### 3.2 AIV：UB 内反量化累加

AIV 按行 tile 读取各 K-group 的 int32 partial 和 channel scale，在 UB 内完成：

```text
int32 partial
  -> Cast(fp32)
  -> Mul(channel_scale)
  -> Axpy(accum, partial, token_scale)
```

每个 channel scale 在一个 tile 内只搬入一次并供所有行复用。`Axpy` 将原来的
`Muls + Add` 合为一条向量操作。

### 3.3 SwiGLU/Quant 双缓冲与动态量化

累加结果不写回 GM，直接在 UB 内进入 SwiGLU。`N>=1024` 时后处理使用两行静态
ping/pong：相邻两行的 SwiGLU 分别写入 `post0` 和 `post1`，两个结果就绪后再依次进入
ReduceMax 和量化。奇数尾行只使用 `post0`。

```text
row 2i:     accum -> SwiGLU -> post0 --+-> Quant
row 2i + 1: accum -> SwiGLU -> post1 --+-> Quant
```

UB workspace 明确拆分为：

```text
post0 | post1 | abs | reduceWork | reduceResult(32B) | reduceTmp(32B)
```

`post0/post1` 与 ReduceMax workspace 不发生别名。量化直接消费 post buffer，双缓冲路径不再
执行旧实现中每行一次的 UB-to-UB DataCopy。`N<1024` 保留单行路径，避免小 shape 的
ping/pong 固定调度开销。

随后使用专用 ReduceMax helper：

```text
SwiGLU -> Abs -> ReduceMax -> scale -> FP32 RINT -> Cast/Pack int8
```

量化使用 vector Cast 模板，不使用逐元素 scalar `GetValue/SetValue`。为保持与 PyTorch
reference 的 ties-to-even 行为一致，先在 FP32 域执行 RINT，再进行 int8 packing。

最终 int8 output 和 scale 通过 queue/DataCopy 批量写回 GM。

### 3.4 行 tiling 与多 AIV

后处理按约 160 KiB UB 预算计算最大 tile，并进一步将每次 int32 partial CopyIn 控制在
约 16 KiB：

```text
parallel_tile_rows = 16 KiB / (N * sizeof(int32))
tile_rows = min(UB_capacity_rows, parallel_tile_rows)
```

tile 按 `tileIdx = blockIdx + n * aivCoreNum` 分发给多个 AIV。TPipe buffer 按最大 tile
容量每个 expert 初始化一次，尾 tile 只调整有效数据长度，不重复 Reset/InitBuffer。

### 3.5 Workspace 布局

partial workspace 使用精确容量：

```text
G * M * N * sizeof(int32)
```

不能使用 `G * aicNum * baseM * baseN` 近似。后者在 `K=N=4096、G=32` 时不足，曾导致
MTE DDR 地址越界。Host tiling 与 kernel 当前使用完全相同的容量公式。

## 4. 主要优化路径

以下性能均来自 Ascend 910B、CANN 8.5.1。小矩阵代表形状为：

```text
M=257, K=512, N=256, experts=4
group_size=128, scale_dtype=fp32
warmup=10, iterations=100
```

| 阶段 | 主要修改 | 延迟 | 相对上一步 | 相对初始 |
|---|---|---:|---:|---:|
| 初始正确性路径 | scalar/debug 风格后处理 | 0.213 ms | - | - |
| X 只打包一次 | 移出 expert 循环 | 0.197 ms | -7.5% | -7.5% |
| 多 AIV tile 并行 | 后处理不再只使用单 AIV | 0.164 ms | -16.8% | -23.0% |
| 16 KiB partial tile | 增加 tile 数，提高 AIV 并行度 | 0.100 ms | -39.0% | -53.1% |
| scale packing 向量化 | 移除逐 channel scalar 循环 | 0.097 ms | -3.0% | -54.5% |
| Cube 直读原始 NZ weight | 取消 weight packing | 0.091 ms | -6.2% | -57.3% |
| 同步精简 | 删除冗余全核同步 | 0.080 ms | -12.1% | -62.4% |
| Axpy 与初始化优化 | 融合 Muls/Add；每 expert 初始化 | 0.078 ms | -2.5% | -63.4% |
| SwiGLU/Quant 双缓冲 | N>=1024 两行 ping/pong；小 N 保留单行路径 | 0.081 ms | +3.8% | -62.0% |

说明：表中的百分比按对应阶段代表测量值计算；板卡状态会造成轻微波动，应以多轮中位数为准。

### 4.1 已保留的优化

1. X packing 从每 expert 一次改为每次 invocation 一次。
2. scale packing 从 scalar GetValue/SetValue 改为 DataCopyPad + vector Cast。
3. weight 不再打包，Cube 直接读取原始 NZ tensor。
4. AIC 遍历完整 M/N blocks，AIV 并行处理独立行 tile。
5. 按 UB 容量和 16 KiB CopyIn 目标联合选择 tileRows。
6. partial、反量化、SwiGLU、ReduceMax 和量化在 UB 内连续完成。
7. 使用 `Axpy(accum, partial, tokenScale)` 合并 Muls 和 Add。
8. TPipe Reset/InitBuffer 从每 tile 降为每 expert 一次。
9. 删除 Cube 前和最终循环后的冗余 `SyncAll`。
10. `N>=1024` 使用两行 SwiGLU/Quant ping/pong，post/abs/reduce workspace 显式隔离。
11. 对常见 `tileM<=4` 路径将 per-token scale 的 GM scalar 读取和 V/S 硬同步移出
    K-group 循环，并批量发射多行 Mul/Axpy；较大 tile 保留原始路径回退。

### 4.2 实验后未保留的方案

- **token scale 批量 DMA**：仍需逐行同步才能保证正确，实测约 `0.081-0.083 ms`，慢于
  scalar GM 读取版本，因此撤回。
- **删除反量化循环中的 V_S**：导致 82/257 行 scale 错误，最大绝对误差约 8411，证明该
  硬件事件不能直接删除。
- **动态 LocalTensor ping-pong 双缓冲**：CANN 8.5.1 编译器无法生成对应 AIC/AIV 对象。
  若后续重试，应使用显式静态分支或模板展开。

### 4.3 2026-07 GMM1 标量同步与 K-group 优化

Ascend 910B 上 AIC Cube 和 AIV 后处理属于不同核，AIC 的本地 L1/UB 不能直接作为
AIV 输入，因此当前 `MatmulApiTiling + IterateAll` ABI 仍需通过 GM 传递 partial。此次优化
在不改变 ABI 的前提下做了两层收敛：

1. token scale 在一个 row tile 内只读取一次，V->S/S->V 同步从每个
   `(K-group,row)` 一次降为每 tile 一次；真实 `M=8192,K=2048,N=1536,E=128,group=256`
   延迟由 `1.667 ms` 降至 `1.586 ms`，提升约 `4.9%`。
2. Qwen3-30B-A3B 的 GMM1 使用 full-K group 2048，将 partial/FixPipe 轮数从 8 降为 1；
   TP-local GMM2 仍通过 gcd fallback 使用 group 128。

相同 batch6/ACL Graph 的端到端结果：

| GMM1 group | K-group 数 | 6-window PPL | 吞吐 |
|---:|---:|---:|---:|
| 256 | 8 | 8.263499 | 4434.70 tok/s |
| 512 | 4 | 8.247595 | 4649.06 tok/s |
| 1024 | 2 | 8.223013 | 5095.63 tok/s |
| 2048 | 1 | 8.246581 | 5866.54 tok/s |

20-window 精度复核中，group1024/group2048 PPL 分别为 `8.656323/8.656475`；差值
`0.000152`。group2048 吞吐为 `5790.84 tok/s`，group1024 为 `5120.75 tok/s`。
如需更保守的量化粒度，可通过 `GROUP_SIZE=1024` 或 `GROUP_SIZE=256` 回退。

### 4.4 full-K profiling 与 N/M tiling 复核

对相同 `M=8192,K=2048,N=1536,E=128` shape 采集 7 组标准 `msprof` 指标后，
group2048 已经消除了 group256 的主要 FixPipe/标量热点：

| GMM1 group | Task duration | AIC MTE2 | AIC FixPipe | AIC scalar |
|---:|---:|---:|---:|---:|
| 256 | 2416.788 us | 36.7% | 44.1% | 48.4% |
| 2048 | 540.631 us | 49.1% | 20.7% | 19.7% |

group2048 下 AIC/AIV 时间分别为 `539.016/537.828 us`，20 个 AIC 的负载不均衡仅
`0.15%`，头开销 `0.3%`。当前主热点已经转为 Cube 侧 MTE2/MTE1 数据搬运，而不是
AIV 标量计算或核间调度。原始解析结果归档在 `docs/perf/round_001`；group256 对照位于
`docs/perf/round_002`。

本轮放宽了 `GROUP_QUANT_GMM1_SINGLE_N` 的 host 上限到 2048，并将默认值从 512 调整为
768：在保留 workspace 预拷贝/L2 预热的情况下，N=1536 恰好切成两个均匀任务，6-window
端到端吞吐从 `5866.54` 提升到 `5994.65 tok/s`（约 `2.18%`），PPL 保持 `8.246581`。

以下变体因端到端退化、微基准退化或精度失败而未保留：

- `singleN=768` 配合 full-K 直读原始 X 时，端到端吞吐只有 `5803.88 tok/s`；因此只保留
  768 分块，恢复 workspace 预拷贝/L2 预热。
- full-K X 合并为 32 KiB 大块预拷贝后，M=8192/M=12288 分别为 `0.452/0.478 ms`，
  没有优于逐行并行预拷贝。
- `basicM=64` 在 M=8192 时为 `0.460 ms`，M=12288 时退化到 `0.791 ms`，已恢复 128。
- `doMTE2Preload=1` 与当前动态 expert tile/手工 step-depth 配置不兼容，首个精度用例
  即出现 63/64 个 scale 不一致，未进入性能测试。

恢复后的 full-K、singleN=512 对照在 400 次微基准中为 `0.444 ms`；默认 singleN=768
在端到端验证中 quant/PPL 保持一致。

### 4.5 双缓冲实测结论

双缓冲已经进入生产路径，但现阶段主要价值是建立稳定的两行软件流水结构，不应把它描述成
已经带来显著加速：

- 小矩阵由 shape 分支继续走单行路径；三轮为 `0.083/0.081/0.081 ms`，中位数
  `0.081 ms`，相对双缓冲前最佳测量 `0.078 ms` 有约 3.8% 波动/回退。
- `M=257,K=N=4096` 双缓冲后为 `1.168 ms`，双缓冲前为 `1.166 ms`，差异约 0.2%，
  基本性能中性。
- `M=17,K=512,N=1024,experts=4,group_size=64,fp16` 覆盖了双缓冲、奇数尾行、多个
  expert 和多个 K-group，quant 完全相等且 scale 误差为 0。

由于 SwiGLU、Abs、ReduceMax 和 Cast 主要都运行在 Vector pipeline，两行 UB ping/pong
本身不能像 MTE/Vector 双缓冲一样天然形成跨执行域重叠。后续若要获得明确收益，应结合
profiling 将 tile CopyOut/下一 tile 计算做 MTE3/Vector overlap，而不是继续增加 post buffer。

## 5. 当前性能与正确性

### 5.1 代表小矩阵

```text
M=257, K=512, N=256, experts=4, group_size=128, fp32
quant_output_equal=True
scale_mean_abs_diff=0
scale_max_abs_diff=0
avg_latency_median=0.081 ms
throughput约3.17M tokens/s
```

### 5.2 K=N=4096 大矩阵

```text
M=257, K=4096, N=4096, experts=4
group_size=128, group_num=32, fp32
warmup=10, iterations=100

quant_output_equal=True
scale_mean_abs_diff=0
scale_max_abs_diff=0
avg_latency=1.168 ms
throughput=220,119 tokens/s
effective_tops=7.386
```

### 5.3 已覆盖形状

精度回归已覆盖：

- `M=1/2/16/256/257`；
- 单 expert 和多 expert；
- `N=256/1024/4096`；
- `K=512/4096`；
- `group_size=64/128`，并验证过原始权重直读的 `group_size=512`；
- weight scale `fp32/fp16`；
- M 尾 tile 和多个 N block。

已支持用例均要求 `quant_output_equal=True`，并且 output scale 与 Python reference 的绝对误差为 0。

## 6. 构建与测试

### 6.1 构建安装

```bash
cd /path/to/vllm-smooth
source env_vllm_ascend.sh
GROUP_QUANT_STAGE=8 custom_op/scripts/build_groupquant_gmm_swiglu.sh
source custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash
```

### 6.2 代表正确性与性能测试

```bash
python -m tests.test_groupquant_gmm_swiglu_op \
  --m 257 --k 512 --n 256 --experts 4 \
  --group-size 128 --scale-dtype fp32 \
  --warmup 10 --iters 100
```

大矩阵：

```bash
python -m tests.test_groupquant_gmm_swiglu_op \
  --m 257 --k 4096 --n 4096 --experts 4 \
  --group-size 128 --scale-dtype fp32 \
  --warmup 10 --iters 100
```

系统测试：

```bash
tests/custom_op/grouped_matmul_swiglu_quant_weight_nz_tensor_list/run.sh
```

## 7. 源码结构

```text
quant_matmul/
├── README.md
└── KernelLaunch/                         # 独立直调参考实现

custom_op/grouped_matmul_swiglu_quant_weight_nz_tensor_list/
├── op_kernel/
│   ├── grouped_matmul_swiglu_quant_weight_nz_tensor_list.h
│   └── grouped_matmul_swiglu_quant_weight_nz_tensor_list_utils.h
├── op_host/
│   ├── grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.cpp
│   └── op_api/
└── tests/st/
```

生产入口为 `ProcessGroupQuantProduction()`。历史 stage9-stage57 调试代码保留在未调用的
`ProcessGroupQuantLegacyDebug()`，不属于当前生产调用图。

更完整的精度定位、硬件同步陷阱和每轮实验记录见仓库根目录的
`stage56_debug_notes.md`。

## 8. 硬件注意事项

Ascend 910B 上 scalar、Vector、MTE 和 Cube 属于不同执行域，跨域可见性必须使用正确事件：

- DMA 写 UB 后由 scalar 读取：`MTE2_S`；
- Vector 写 UB 后由 scalar 读取：`V_S`；
- scalar 结果作为 Vector 标量参数：`S_V`；
- scalar 写 UB 后由 MTE3 搬出：`S_MTE3`。

不要使用 scalar `GetValue` 读取 MTE/Cube 刚写入的 GM 作为正确性 probe；该读取可能得到 0
或 stale value。调试中间结果应优先使用 Vector/MTE 路径批量搬出。
