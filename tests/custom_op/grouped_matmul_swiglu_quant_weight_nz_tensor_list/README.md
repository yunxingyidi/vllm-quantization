# GroupQuant 融合 MoE SwiGLU ST

这些脚本运行当前 PyTorch/torch_npu 回归用例。请先从仓库根目录构建 registry
算子并加载环境：

```bash
cmake -S custom_op -B custom_op/build \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
cmake --build custom_op/build --target registry_ops
source env_vllm_ascend.sh
tests/custom_op/grouped_matmul_swiglu_quant_weight_nz_tensor_list/run.sh
```

用例覆盖 K-split、单 group、FP16/FP32 scale、空 expert、多行输入和非整 tile。
日志默认写入 `tests/artifacts/grouped_matmul_swiglu_quant_weight_nz_tensor_list/`，
也可通过 `LOG_DIR` 指定其他位置。
