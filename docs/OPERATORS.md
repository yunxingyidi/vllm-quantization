# GroupQuant 算子说明

## 算子关系

| 算子 | 角色 | 默认路径 | 输出 |
|---|---|---|---|
| `Gmm1VecinFullk` | GMM1 + 反量化 + SwiGLU + 动态量化 | 多组或原生契约不满足时 | INT8 `[M,N/2]` + FP32 `[M]` |
| `npu_grouped_matmul_swiglu_quant` | 原生融合GMM1 | 单组且满足原生条件时 | INT8 H + FP32 token scale |
| `QuantMatmulExpertKGrouped` | 全 expert GMM2 | 是 | BF16 `[M,N]` |
| `GroupedMatmulSwiGluQuantWeightNzTensorList` | 通用 GMM1 回退 | shape/版本不满足时 | INT8 + FP32 scale |
| `quant_group_matmul` | 逐 expert/Linear 正确性回退 | 否，不进生产包 | FP16/BF16 |

## 原生单组 GMM1

默认自动选择：K=2048 单组、N 为 64 倍数且不超过 1536、启用融合分支、
无非零 swiglu_limit、无动态 EPLB，且本机提供原生接口。
加载期保留 `[E,K,N]` 完整 NZ INT8 权重及 `[E,N]` FP32 channel scale，
forward 调用 `torch_npu.npu_grouped_matmul_swiglu_quant`，输出 INT8 H 和
FP32 token scale，直接连接双 AIV GMM2。没有 forward repack。
此后端不承诺复现自定义 FixPipe FP16 中间舍入；精度和 PPL 需按所选后端重新验证。
`GROUP_QUANT_GMM1_BACKEND=custom` 在加载期选择自定义布局；切换需重启模型。

## Gmm1VecinFullk

输入为路由阶段已经量化的 INT8 `[M,2048]`、per-token FP32 scale、每 expert 的
INT8 NZ 权重、打包 INT64 weight scale 和累计 expert token 数。权重逻辑宽度
`local_N` 必须是 64 的倍数且不超过 1536；K-group 数必须整除 2048，group size
需 16 对齐。输出宽度为 `local_N/2`。

该 MIX kernel 让 AIC 完成 INT8 MatMul/FixPipe，配对 AIV 在本地完成 FP32 分组累加、
SwiGLU 和 per-token 量化，不把 INT32 中间结果写回 GM。生产模型在加载权重时一次性
生成 packed scale，forward 不应再调用 `npu_trans_quant_param`。

多组且 N<=768、半宽为 64 的倍数时，默认使用 M32 上限、4096-float
partial scratch 和完整 FP32 NZ accumulator，沿用 NZ epilogue，最后仅将
半宽 SwiGLU 结果转 ND。FP16 FixPipe 舍入与 FP32 group 累加顺序保持一致；
其他宽度自动回退。`GROUP_QUANT_GMM1_GROUP_NZ=0` 可在新进程中恢复上一轮
M16 累加/逐行 epilogue；再设 `GROUP_QUANT_GMM1_NZ_ACCUM=0` 才回到更早的
逐组 ND scatter。

满足 grouped NZ epilogue 条件时，还默认在一次 kernel 调用内复用未变化的
Matmul OrgShape；M/N 尾块变化时重新设置，各组 SingleShape、A/B 地址和独立
scale 继续更新。`GROUP_QUANT_GMM1_REUSE_ORG_SHAPE=0` 可在新进程中关闭该
优化作对照。UB/权重布局不变。

## QuantMatmulExpertKGrouped

```text
x_int8          [G*M, group_size]
weight_nz_int8  expert-major NZ packed
group_list      [E]，每 expert token 数（非累计）
weight_scale    [E*G, N] FP32
x_scale         [M] FP32
output          [M, N] BF16
```

也支持连续 INT8 `[M,G,group_size]`，rank3 明确表示原 ND 激活布局，
无需生成 group-major 副本。多组预量化激活默认使用该路径；单组和动态
量化输入保留原契约。`GROUP_QUANT_GMM2_ND_INPUT=0` 在新进程中恢复旧
输入重排作对照。实验 local-int32 模板仅支持旧契约。
精度与性能使用tests/custom_op下对应测试重新验证。

算子一次调度全部 expert。Qwen3-30B-A3B TP=2 的默认参数为 20 核、
`baseM/baseN/baseK=64/512/128`、4 个 pipeline slot、完整本地 `K=384` 单组。
环境参数集中在 `env_vllm_ascend.sh`。

## 通用 GMM1 回退

`GroupedMatmulSwiGluQuantWeightNzTensorList` 支持更一般的 K/N 和动态 tensor list，
但需要更大的 partial workspace。它保证主算子 dispatch guard 不满足时仍可执行，
不应在目标 Qwen3 shape 上成为常态。整网日志出现回退时应先检查 adapter、packed
scale dtype/shape 和 `GROUP_QUANT_USE_GMM1_VECIN_FULLK`。

## 构建与加载

三个 registry 算子必须在同一次 CANN 构建中生成一个 vendor：

```bash
bash custom_op/scripts/build_groupquant_moe_fused.sh
source custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash
```

GMM1 与 GMM2 的独立 Torch adapter 仅负责注册 `torch.ops._C_ascend.*` 和调用 ACLNN；
真正 op-api 从 `GROUP_QUANT_CUSTOM_OPAPI_PATH` 指向的统一库解析。不要把单算子生成的
另一个 `libcust_opapi.so` 混入 `LD_LIBRARY_PATH`。
