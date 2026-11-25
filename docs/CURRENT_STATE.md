# 当前代码状态（整理于2026-11-25）

本次整理未更改推理算子、量化算法或默认参数，也未重新运行性能测试。下面按现有源码确认默认行为。

## 生产入口与数据流

入口run.sh → ppl_eval.py → Smoothquant/custom_moe_quant.py：仅MoE权重及激活W8A8，普通线性层仍为BF16。权重在加载阶段在线量化并准备NZ/FixPipe等布局；激活由运行时按token动态量化。

- 默认GROUP_SIZE=2048；GMM1在符合原生契约的单组场景使用原生融合GMM+SwiGLU+量化；多组及不符合条件时使用自定义Gmm1VecinFullk。
- GMM2默认GROUP_QUANT_GMM2_GROUP_SIZE=0，即每张卡本地完整K维一组；运行QuantMatmulExpertKGrouped。双AIV、ND输入及多组相关优化由现有参数和形状条件选择。
- 默认路径为routing内量化 → GMM1 → GMM2 → combine。GROUP_QUANT_DISPATCH_SCATTER=1为显式实验分支，默认0；支持条件内融合原token量化及top-k scatter，不满足则回退。
- run.sh默认TP2、batch6、100个2048-token WikiText2窗口；它是PPL回归入口，不是完整在线服务吞吐入口。单组不表示整个模型只量化一个专家。
- 测试入口的GROUP_QUANT_STAGE_BF16_ON_CPU=1仅用于资源对照时缓解BF16源权重与INT8缓冲并存的初始化OOM，没有成为生产默认；推理参数仍驻留NPU。

## 核心代码地图

| 位置 | 职责 |
|---|---|
| Smoothquant/custom_moe_quant.py | 权重量化、GMM后端及布局选择 |
| Smoothquant/fuse_moe/moe_comm_method.py | 通信/路由分发适配 |
| Smoothquant/fuse_moe/dispatch_quant_scatter.py | 可选量化分发融合及回退条件 |
| Smoothquant/fuse_moe/moe_mlp.py | 专家GMM计算与调用契约 |
| custom_op/gmm1_vecin_fullk/ | 自定义多K组GMM1 |
| custom_op/quant_matmul_expert_k_grouped/ | expert-aware GMM2 |
| custom_op/dispatch_quant_scatter/ | 原token量化与分发融合 |
| tests/performance/ | 配对测试、采集与汇总 |

## 验证与输出

当前测试入口见[测试说明](../tests/performance/README.md)。历史原始结果和统计已清空；后续性能与质量结论需要重新测试。测试结果写入tests/artifacts，分析脚本可按需重新创建docs/performance_analysis。

已交付Word及配图保留在[年度总结目录](reports/年度总结_2026/)，仅作历史汇报材料，不代表当前代码重新验证。旧设计提案和分阶段性能报告已删除，当前实现以源码及本页默认行为为准。
