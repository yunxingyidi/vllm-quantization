# 项目结构与 vLLM-Ascend 集成指南

## 1. 交付范围

本项目以 MoE-only W8A8 路径作为生产入口：普通 Linear 不量化，
MoE 权重在线转换为 INT8 GroupQuant 状态。RouteIndex/ByPerm 原型、离线 VQ、激活
scale 校准、profiling 数据库和历史实验实现均不属于集成范围。

基准环境为 Ascend 910B3、CANN 8.5.1、Qwen3-30B-A3B

## 2. 推理数据流

```text
HF BF16 expert weights
  -> GroupQuantMoEMethod.process_weights_after_loading
  -> {weight_q, delta_base, exponent, packed NZ/FixPipe scale}

hidden states + router logits
  -> vLLM-Ascend select_experts
  -> vLLM-Ascend token_dispatch(with_quant=True)
  -> INT8 routed tokens + FP32 per-token scale + expert group list
  -> Gmm1VecinFullk
  -> INT8 SwiGLU output + FP32 per-token scale
  -> QuantMatmulExpertKGrouped
  -> BF16 routed output
  -> vLLM-Ascend token_combine
```

路由、通信和 combine 仍由 vLLM-Ascend 管理。本项目只扩展 quantization config、
GroupQuant 权重状态以及 dispatch/combine 中间的 MLP 计算，因此与其他
vLLM-Ascend 修改集成时冲突面集中在 MoE quant method 和 comm method 两处。

## 3. 文件职责

| 文件/目录 | 职责 | 集成要求 |
|---|---|---|
| `Smoothquant/group_quant_config.py` | 注册 `group_quant`，为 FusedMoE 选择量化方法 | 必需 |
| `Smoothquant/custom_moe_quant.py` | 创建/量化权重，适配 select_experts、EPLB 和新旧 ABI | 必需 |
| `Smoothquant/group_quant_moe_scheme.py` | GroupQuant MoE 权重数学与元数据结构 | 必需 |
| `Smoothquant/fuse_moe/moe_comm_method.py` | 在原生 dispatch/combine 之间注入 GroupQuant MLP | 必需 |
| `Smoothquant/fuse_moe/moe_mlp.py` | GMM1/GMM2 选择、布局转换、adapter 懒加载 | 必需 |
| `moe_runtime_args.py`, `moe_stage_contracts.py` | vLLM-Ascend typed MoE ABI 兼容 | 必需 |
| `custom_linear_quant.py`, `group_quant_linear_scheme.py` | 共用量化/打包函数；兼容可选 Linear | 保留，但默认不启用 Linear |
| `custom_op/` 三个 registry 算子 | 生产计算内核 | 必需 |
| `ppl_eval.py`, `ppl_data.py`, `run.sh` | PPL 回归 | 建议保留 |

## 4. 集成方式 A：低冲突运行时插件

适合先与另一个 vLLM-Ascend 分支联调：

1. 将本仓加入 `PYTHONPATH`，或把 `Smoothquant/` 作为 Python 包安装到目标环境。
2. 用目标分支自身的 `csrc` 执行
   `VLLM_ASCEND_CSRC=... bash custom_op/scripts/build_groupquant_moe_fused.sh`。
3. 启动前 source `env_vllm_ascend.sh`。它会加载本仓 vendor，并把其路径放在
   vLLM-Ascend 自带 OPP 之前。
4. 在创建 `LLM` 前 import `Smoothquant.group_quant_config`，并确保平台允许
   quantization 名称 `group_quant`。`ppl_eval.py` 中的
   `maybe_register_custom_quant_backend()` 是
   可直接迁移的最小注册示例。
5. 使用参数：`quantization=group_quant`、`quantize_moe=True`、
   `quantize_linear=False`、`residual_group=2048`。

## 5. 集成方式 B：合入 vLLM-Ascend 源码树

稳定后建议原生化，按以下顺序合并：

1. 将 `GroupQuantConfig` 加入 vLLM quantization registry 和 Ascend 平台支持列表。
2. 将 `GroupQuantFusedMoEAdapter`/`GroupQuantMoEMethod` 放入 quantization 模块，
   保留权重名和 `process_weights_after_loading()` 的打包逻辑。
3. 不整体覆盖目标分支的 `moe_comm_method.py`；只把 `group_quant` 元数据字段和
   `group_quant_apply_mlp()` 分支合到目标分支现有 typed/legacy ABI。目标分支的
   dispatch、MC2、EPLB 和 combine 修改优先。
4. 将三个算子目录加入上游 `csrc/gmm/` 的算子列表；确保它们打入同一个 vendor
   包和同一个 `libcust_opapi.so`。
5. 将两个 adapter 的注册代码接入上游扩展，或继续独立编译。若独立编译，必须
   保留 `GROUP_QUANT_CUSTOM_OPAPI_PATH` 的绝对路径解析，以避免同 SONAME 冲突。
6. 合并后先跑 Python reference，再跑算子 ST，最后跑 1-window 和 100-window PPL。

### 易冲突点

- `MoEFusedExpertsInput` 字段随 vLLM-Ascend 版本变化；本仓用签名检查兼容，不应把
  某一版本的完整 dataclass 强行覆盖到目标分支。
- EPLB 搬运必须同时移动每个 expert 的 `weight_q/delta_base/e`，不能只移动 INT8。
- GMM1/GMM2 adapter 必须解析同一 vendor 的 op-api，不能各自安装同名库。
- GMM1 使用累计 expert token 数，GMM2 使用每 expert token 数。
- `residual_group=2048` 作用于 GMM1；GMM2 通过 `GROUP_QUANT_GMM2_GROUP_SIZE=0`
  使用完整 TP-local K（目标配置为 384）。

## 6. 验证门槛

```bash
bash custom_op/scripts/build_groupquant_moe_fused.sh
MAX_WINDOWS=1 MODEL_PATH=/path/to/model bash run.sh
PPL_RESULT_PATH=/tmp/groupquant_100w.json \
  MAX_WINDOWS=100 MODEL_PATH=/path/to/model bash run.sh
```

有本机测试脚本时，可运行 `python -m tests.test_groupquant_moe_reference` 和
`python -m tests.test_groupquant_moe_twostage` 检查 MoE 参考实现和两阶段路径。

验收要求：测试通过、无 fallback warning、PPL 与保存结果一致、吞吐以至少两次均值
比较。先固定模型、数据、TP、batch、window 数和 ACL Graph，再判断性能变化。

## 7. 构建产物边界

`custom_op/build_root/`、各算子 `build/`/`out/`、`tests/artifacts/profiles/` 都是可重建
产物，不应提交或复制到集成分支。需要交付的只有源码、脚本、CANN 兼容 patch 和
三份 PPL JSON。
