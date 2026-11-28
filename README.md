# GroupQuant for vLLM-Ascend

面向Qwen3-30B-A3B、Ascend 910B 和 910C 的 MoE W8A8 量化与算子优化工程。权重按 K 维分组量化，激活按 token 动态量化。Attention、Router、LM head 等非 MoE 线性层保持 BF16。

```text
路由分发（含per-token INT8量化）
  → 单组原生融合GMM1 / 多组自定义GMM1
  → expert-aware INT8 GMM2
  → token combine
```

## 使用

```bash
export VLLM_ASCEND_CSRC=/path/to/vllm-ascend/csrc
bash custom_op/scripts/build_groupquant_moe_fused.sh
ASCEND_RT_VISIBLE_DEVICES=0,1 MODEL_PATH=/path/to/Qwen3-30B-A3B bash run.sh
```

设备编号按实际可用卡设置。环境路径可通过CONDA_ENV_NAME、CANN_HOME、ATB_HOME、ASDSIP_HOME覆盖；数据集路径通过WIKITEXT2_DATASET_DIR覆盖。

构建脚本按当前 NPU 选择 `ascend910b` 或 `ascend910_93`，也可用 `GROUP_QUANT_SOC` 指定目标。使用 vLLM-Ascend 0.19.1rc1 wheel 时，`VLLM_ASCEND_CSRC` 指向支持 `--pkg --ops --soc` 的算子打包源码，`VLLM_ASCEND_ADAPTER_CSRC` 指向对应 0.19.1rc1 的 `csrc`，用于编译 PyTorch adapter。本机的 910C 构建使用打包源码提交 `bd42eea01d8312c4df469fabaed45f77a12df4d3`。

`run.sh` 会加载仓库内安装的自定义算子包。本机使用 `vllmq`、`~/Ascend/cann`（CANN 8.5.2）以及 `/data/models/Qwen/Qwen3-30B-A3B`。未设置本地 WikiText-2 路径时，从 `Salesforce/wikitext` 加载测试集。

run.sh默认TP2、batch6、100个2048-token窗口，用于PPL回归。显存利用率默认为0.90，给TP收集prompt logprobs预留空间，可通过`GPU_MEMORY_UTILIZATION`覆盖。其离线prompt-logprob计时不能当作在线decode吞吐。默认GMM1单组K2048，GMM2为每卡本地完整K一组；多组示例：

```bash
GROUP_SIZE=128 GROUP_QUANT_GMM2_GROUP_SIZE=128 bash run.sh
```

实际分组受本地K及兼容约束影响。单组满足条件时GMM1自动使用原生融合算子；多组等场景回退自定义实现。GMM2采用双AIV/ND输入等现有优化。量化分发scatter仍为显式实验选项（默认关闭）。

## 文档和代码入口

- [当前代码状态与默认行为](docs/CURRENT_STATE.md)
- [集成指南](docs/INTEGRATION.md)、[算子与布局契约](docs/OPERATORS.md)
- [性能、PPL和诊断脚本索引](tests/performance/README.md)
- [年度总结Word与配图](docs/reports/年度总结_2026/)

生产代码在Smoothquant/与custom_op/，测试输出写入tests/artifacts/（历史结果已清空）。分析脚本按需创建docs/performance_analysis/作为输出目录。2026-11-24已清理重复图表、临时日志及导出结果，未修改推理算法或重测性能。

算子专项加速不等同于整网加速。BF16/单组/多组、不同TP和负载的结果需按历史测试条件比较。
