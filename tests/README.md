# 测试目录

| 位置 | 用途 |
|---|---|
| test_*.py、conftest.py | GroupQuant数学参考、动态量化、部分和及MoE回归 |
| custom_op/ | GMM1/GMM2精度、ND输入、多组路径测试和微基准 |
| performance/ | PPL、HTTP吞吐、显存、设备时间线及激活诊断 |
| scripts/ | PPL工作负载profiling辅助入口 |
| ppl_data.py | PPL/真实文本诊断共用数据加载 |
| artifacts/ | 后续测试输出目录；历史结果已全部清空 |

2026-11-24清理了旧Profiler明细、数据库、日志、tensor、__pycache__和测试build产物。随后再次清空artifacts历史结果，仅保留测试源码和输出说明。

复测入口见performance/README.md，旧报告生成脚本已删除。算子测试run.sh会重新构建所需二进制。NPU测试需先加载env_vllm_ascend.sh并配置可用设备；本次整理没有运行NPU性能回归。
