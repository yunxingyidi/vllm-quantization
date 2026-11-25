# 当前性能与诊断入口

在仓库根目录source env_vllm_ascend.sh后运行。部分入口仍有默认模型、设备及带日期的输出路径，使用前按实际环境配置。历史实验结果已清空，不能直接运行汇总脚本恢复旧结论。

| 目标 | 保留入口 | 说明 |
|---|---|---|
| BF16/W8A8完整WikiText2 PPL | run_ppl_bf16_compare.sh | 默认TP2、物理4/5卡；PPL不是任务准确率 |
| 官方vLLM HTTP推理对比 | run_vllm_serve_compare.sh / .py | VLLM_SERVE_BENCH_SUITE=full跑完整四场景矩阵 |
| HTTP完整矩阵汇总 | summarize_vllm_full_suite.py <结果目录> | 验证请求成功数、输入输出长度及两轮指标 |
| 显存、AICore忙碌比例、吞吐、TTFT、独立时间线 | run_resource_compare.py | --devices / --modes / --output；TP1量化加载需按入口启用CPU暂存 |
| 资源汇总 | summarize_resource_compare.py <结果目录> | 正常压测与独立profile分开统计 |
| Profiler离线解析 | analyse_prefill_profiles.py <profile目录> | 输入必须是新采集的完整数据 |
| GMM1/GMM2微基准及精度 | ../custom_op/下对应脚本 | profile_operator_pair.py提供共享硬件计数采集函数，被两个benchmark导入 |
| 量化分发微基准 | benchmark_dispatch_quant.py、benchmark_dispatch_scatter.py | scatter依赖前者的输入和路由辅助函数，两个都保留 |
| 真实激活采样与INT8误差图 | capture_activation_outliers.py、analyse_activation_outliers.py | 默认NPU1、7层×8窗，需先采集再分析；不是整网W8A8精度测试 |
| 加载期CPU暂存验证 | check_cpu_staging.py | resource_probe.py提供对应测试挂钩 |

基础依赖：vllm_cli_with_groupquant.py注册GroupQuant并启动官方CLI；resource_probe.py提供资源记录及RPC；profiler_patch/sitecustomize.py配置采集。这些文件不是重复入口，删除会破坏现有测试。

2026-11-24已删除12个历史实验脚本及整个tests/reporting。性能以官方HTTP入口为主，精度以PPL及算子正确性测试为主。旧离线/稳态筛选结果不再提供专用复现入口；旧Word及配图仍在docs/reports/年度总结_2026。
