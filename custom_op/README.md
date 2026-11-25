# GroupQuant 自定义算子

生产包由 `scripts/build_groupquant_moe_fused.sh` 一次构建，避免多个同名
`libcust_opapi.so` 互相覆盖：

- `gmm1_vecin_fullk/`：多组或原生条件不满足时的自定义GMM1，输出INT8 + scale。单组满足条件时优先调用torch_npu原生融合GMM1。
- `quant_matmul_expert_k_grouped/`：主路径 GMM2，一次处理全部 expert。
- `grouped_matmul_swiglu_quant_weight_nz_tensor_list/`：通用 GMM1 回退。
- `quant_group_matmul/`：可选的逐 expert/线性层正确性回退，不进入默认构建。

```bash
export VLLM_ASCEND_CSRC=/path/to/vllm-ascend/csrc
bash custom_op/scripts/build_groupquant_moe_fused.sh
source custom_op/build_root/install/vendors/custom_transformer/bin/set_env.bash
```

构建产物全部位于被忽略的 `custom_op/build_root/` 和各算子 `build/` 目录，不是
源码交付的一部分。接口与布局见 `docs/OPERATORS.md`。

可选 dispatch 实验位于 `dispatch_quant_scatter/`，独立构建后以
`GROUP_QUANT_DISPATCH_SCATTER=1` 启用受限的大输入 legacy AllGather 路径。
它不属于默认生产 OPP 构建；见该目录 README 和
`Smoothquant/fuse_moe/dispatch_quant_scatter.py`中的运行条件。

## 清理与运行依赖

2026-11-25已删除build_root/csrc构建副本，约203 MiB，可由prepare_vllm_ascend_csrc.sh从VLLM_ASCEND_CSRC重新准备。源码目录全部仍有构建或运行引用，本次不删除任何算子功能。

注意：build_root/install虽是构建产物，但env_vllm_ascend.sh当前从这里加载OPP/op-api；gmm1_vecin_fullk/build与quant_matmul_expert_k_grouped/build中的Torch适配.so也由运行时代码加载。dispatch_quant_scatter/build中的.so由实验分支加载。因此这些运行文件本次保留，不能将所有build目录直接清空后仍期望无需重建即可推理。
