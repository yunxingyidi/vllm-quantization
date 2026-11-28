# 测试目录

| 位置 | 用途 |
|---|---|
| 顶层 `test_*.py` | 本机 GroupQuant 数学参考、动态量化、部分和及 MoE 回归，由 Git 忽略 |
| `custom_op/` | GMM1/GMM2 精度、ND 输入、多组路径测试和微基准 |
| `performance/` | [本机 PPL 对照、HTTP 吞吐与 profiling](performance/README.md)，脚本由 Git 忽略 |
| `local/` | 本机采样、资源排查和历史复现脚本，由 Git 忽略 |
| `artifacts/` | 本地实验结果、原始 trace、复现实验脚本和汇报图表 |

仓库根目录的 `run.sh` 和 `ppl_eval.py` 提供共享 PPL 入口，使用同目录的 `ppl_data.py`。
算子目录下的 `run.sh` 会准备对应的测试二进制。NPU 测试前先加载 `env_vllm_ascend.sh` 并配置可用设备。
